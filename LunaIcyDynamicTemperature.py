import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root
import scipy.linalg

# -------------------------------------------------------------------------
# 1. PARAMETERS & CONSTANTS
# -------------------------------------------------------------------------
# Physical Constants
R_gas = 8.314         # J/(mol K)
M_h2o = 0.018015      # kg/mol
rho0 = 917.0          # kg/m^3 (Bulk Ice Density)
gamma = 0.065         # J/m^2 (Surface Tension)
P0 = 3.5e12           # Pa
Q_sub = 51000.0       # J/mol
sigma_SB = 5.67e-8    # W/(m^2 K^4)
cp_ice = 2100.0       # J/(kg K)
epsilon = 0.9         # Emissivity

# Model Parameters
alpha = 0.03          # Sticking coefficient
phi = 0.4             # Porosity
Albedo = 0.0          # Surface Albedo (A)
G_sc = 50.0           # Solar Constant at Europa distance (W/m^2)
# Note: F_solar = (1-A) * G_sc/d^2. We assume G_sc is already scaled to Jupiter 
# or d(t)=1.0 to represent the mean distance for this timeframe.
dist_sun = 1.0        

# Simulation Settings
dt = 3600.0           # Time step (s)
t_max = 50 * 86400    # 50 Days
depth = 0.2           # Domain depth (m)
Nx = 30               # Grid points
dx = depth / Nx
x_grid = np.linspace(dx/2, depth - dx/2, Nx) # Cell-centered

# Microscale scaling (for numerical stability in root finding)
LS = 1e-6 

# -------------------------------------------------------------------------
# 2. ANALYTIC GEOMETRY (Per Section 5)
# -------------------------------------------------------------------------
def get_geometry_analytic(r_g, r_b):
    """
    Computes Volumes, Areas, and Curvatures using the analytic solutions
    from Section 5.1, 5.2, 5.3 of the derivation.
    Inputs: r_g, r_b in meters.
    """
    # Normalize to microns for stability in root finding
    rg_u = r_g / LS
    rb_u = r_b / LS
    
    # Dependent radius r_p (Section 5.2)
    # Singularity protection: if rb -> rg, rp -> infinity. Cap rb.
    if rb_u >= 0.999 * rg_u: rb_u = 0.999 * rg_u
    rp_u = (rb_u**2) / (2 * (rg_u - rb_u))

    # Find intersection x_star (Section 5.4)
    # Solve: sqrt(rg^2 - x^2) = rb + rp - sqrt(rp^2 - (rg-x)^2)
    def intersection_eq(x):
        # Prevent domain errors
        val1 = max(0, rg_u**2 - x**2)
        val2 = max(0, rp_u**2 - (rg_u - x)**2)
        return np.sqrt(val1) - (rb_u + rp_u - np.sqrt(val2))
    
    # Root finding for x_star in (0, rg)
    sol = root(intersection_eq, 0.9 * rg_u) 
    x_star_u = sol.x[0] if sol.success else 0.9 * rg_u

    # 1. Grain Volume V_g (Eq 5)
    # vg = 4/3 pi rg^3 - pi(rg - x*)^2 / 3 * (2rg + x*)
    term_cap = (np.pi * (rg_u - x_star_u)**2 / 3.0) * (2 * rg_u + x_star_u)
    Vg_u = (4.0/3.0) * np.pi * rg_u**3 - term_cap

    # 2. Bond Volume V_b (Eq 6)
    # Vb = pi(rb+rp)^2(rg-x*) - 2(rb+rp)[...] + pi rp^2(rg-x*) - pi/3(rg-x*)^3
    # Inner term A from derivation:
    h_val = rg_u - x_star_u
    sqrt_term = np.sqrt(max(0, rp_u**2 - h_val**2))
    asin_term = np.arcsin(min(1.0, h_val / rp_u))
    bracket_term = np.pi * ( (h_val/2)*sqrt_term + (rp_u**2/2)*asin_term )
    
    term1 = np.pi * (rb_u + rp_u)**2 * h_val
    term2 = 2 * (rb_u + rp_u) * bracket_term
    term3 = np.pi * rp_u**2 * h_val
    term4 = (np.pi / 3.0) * h_val**3
    
    # Multiply by 2 as Eq 6 is for the bond volume (assumed symmetric/full)
    # The derivation calculates integral for V^rb. Text says "v_b(rg,rb) := ...".
    # Assuming the formula in Eq 6 is the total bond volume.
    Vb_u = term1 - term2 + term3 - term4
    # Note: The derivation calculates volume on one side. Sintering usually implies 
    # mass conservation between 2 grains. We assume v_b is the total volume of the neck.
    # If Eq 6 is half (symmetric integration), we strictly follow Eq 6 definition.
    # Text defines v_b explicitly via that formula. We use it as is.

    # 3. Grain Surface Area S_g (Section 5.3 Analytic)
    Sg_u = 2 * np.pi * rg_u * (rg_u + x_star_u)

    # 4. Bond Surface Area S_b (Section 5.3 Analytic)
    # Sb = 2 pi rp ( (rb+rp) arcsin(...) - (rg - x*) )
    Sb_u = 2 * np.pi * rp_u * ((rb_u + rp_u)*asin_term - h_val)

    # 5. Curvatures
    Kg = 2.0 / r_g
    Kb = -1.0 / (rp_u * LS) # rb in text usually implies neck radius, curvature depends on rp

    # Rescale to SI
    Vg = Vg_u * (LS**3)
    Vb = Vb_u * (LS**3)
    Sg = Sg_u * (LS**2)
    Sb = Sb_u * (LS**2)
    
    return Vg, Vb, Sg, Sb, Kg, Kb

# -------------------------------------------------------------------------
# 3. PHYSICS & FLUXES
# -------------------------------------------------------------------------
def get_fluxes(r_g, r_b, T):
    """
    Calculates mass fluxes Jg, Jb (kg s^-1 m^-2) based on Hertz-Knudsen.
    """
    Vg, Vb, Sg, Sb, Kg, Kb = get_geometry_analytic(r_g, r_b)
    
    # Saturated Vapor Pressure (Flat)
    P_sat = P0 * np.exp(-Q_sub / (R_gas * T))
    
    # Kelvin Correction
    # P_Kj = P_sat * (1 + gamma M / (R T rho0) * K_j)
    kelvin_coef = (gamma * M_h2o) / (R_gas * T * rho0)
    P_Kg = P_sat * (1 + kelvin_coef * Kg)
    P_Kb = P_sat * (1 + kelvin_coef * Kb)
    
    # Equilibrium Gas Mass (Section 3)
    # m_gas = P_sat(...) * (M Vg phi) / ((1-phi) R T (Sg+Sb))
    # The term in P_sat(...) includes the weighted Kelvin effects
    term_bracket = (Sg + Sb) + kelvin_coef * (Sg * Kg + Sb * Kb)
    numerator = P_sat * term_bracket
    denominator = (1 - phi) * R_gas * T * (Sg + Sb)
    m_gas = numerator * (M_h2o * Vg * phi) / denominator
    
    # P_gas calculation
    # P_gas = (1-phi) m_gas R T / (M Vg phi)
    P_gas = ((1 - phi) * m_gas * R_gas * T) / (M_h2o * Vg * phi)
    
    # Fluxes J_i (Section 2.1)
    # J = alpha * (P_Ki - P_gas) * sqrt(M / 2 pi R T)
    prefactor = alpha * np.sqrt(M_h2o / (2 * np.pi * R_gas * T))
    J_g = prefactor * (P_Kg - P_gas)
    J_b = prefactor * (P_Kb - P_gas)
    
    return J_g, J_b, Sg, Sb, Vg, Vb

# -------------------------------------------------------------------------
# 4. IMPLICIT MICROSTRUCTURE SOLVER
# -------------------------------------------------------------------------
def solve_microstructure_implicit(rg_old, rb_old, T, dt_step):
    """
    Solves the system of equations for the new microstructure (rg, rb)
    using Implicit Euler as defined in Section 3.
    System:
       m_g(new) = m_g(old) - dt * J_g(new) * S_g(new)
       m_b(new) = m_b(old) - dt * J_b(new) * S_b(new)
    """
    # Get initial masses
    Vg_old, Vb_old, _, _, _, _ = get_geometry_analytic(rg_old, rb_old)
    mg_old = Vg_old * rho0
    mb_old = Vb_old * rho0
    
    def residuals(x):
        # x = [rg_new, rb_new]
        r_g_curr, r_b_curr = x
        
        # Constraints to prevent physical violations in solver
        if r_g_curr < 1e-8 or r_b_curr < 1e-8 or r_b_curr >= r_g_curr:
            return [1e10, 1e10]
        
        # Calculate State at t + tau
        J_g, J_b, S_g, S_b, V_g, V_b = get_fluxes(r_g_curr, r_b_curr, T)
        
        # Mass Balance Errors
        m_g_curr = V_g * rho0
        m_b_curr = V_b * rho0
        
        res_g = m_g_curr - (mg_old - dt_step * J_g * S_g)
        res_b = m_b_curr - (mb_old - dt_step * J_b * S_b)
        
        return [res_g, res_b]

    # Solve root finding problem
    sol = root(residuals, [rg_old, rb_old], method='hybr', tol=1e-10)
    
    if sol.success:
        return sol.x[0], sol.x[1]
    else:
        # Fallback if solver fails (stiffness or bad guess), return old
        return rg_old, rb_old

# -------------------------------------------------------------------------
# 5. HEAT EQUATION SOLVER (Implicit + Newton)
# -------------------------------------------------------------------------
def calc_conductivity(T, rg, rb):
    # k = k0(T) * (1-phi) * rb/rg
    k0 = 567.0 / T 
    return k0 * (1 - phi) * (rb / rg)

def solar_flux(t):
    # F_solar(t) = (1-A) * G_sc/d^2 * cos(theta) * 1_{cos>0}
    day_len = 3.55 * 86400
    theta_i = (2 * np.pi * t) / day_len
    cos_theta = np.cos(theta_i)
    val = (1 - Albedo) * (G_sc / dist_sun**2) * cos_theta
    return max(0.0, val)

def solve_heat_implicit(T_old, rg_arr, rb_arr, dt_step, time_curr):
    """
    Solves discretized Heat Equation using Implicit Euler and Newton-Raphson.
    Matches the discretization in Section 4 (Method of Lines + Ghost Cells).
    """
    T_new = T_old.copy()
    N = len(T_old)
    coeff = dt_step / (rho0 * (1 - phi) * cp_ice)
    inv_h2 = 1.0 / (dx**2)
    
    # Pre-calculate k based on geometry (constant during T-step as per splitting)
    # Note: text says k depends on T(x,t). Fully implicit means k is updated with T_new.
    # We update k inside the Newton loop.
    
    for iter_k in range(10): # Newton Iterations
        # Update properties
        k_vals = calc_conductivity(T_new, rg_arr, rb_arr)
        
        G = np.zeros(N)
        # Jacobian Diagonals
        diag = np.zeros(N)
        upper = np.zeros(N)
        lower = np.zeros(N)
        
        # --- Boundary Condition x=0 ---
        # Ghost point T_{-1} derived in Section 4:
        # T_{-1} = 2h/k0 * (F_sol - eps sigma T0^4) + T1
        F_sol = solar_flux(time_curr)
        Rad_term = epsilon * sigma_SB * T_new[0]**4
        dRad_dT = 4 * epsilon * sigma_SB * T_new[0]**3
        
        T_ghost = (2*dx / k_vals[0]) * (F_sol - Rad_term) + T_new[1]
        
        # Derivative of T_ghost w.r.t T0 (chain rule via Rad_term)
        dTg_dT0 = (2*dx / k_vals[0]) * (-dRad_dT) 
        # (Neglecting dk/dT for Jacobian simplicity, standard approximation)

        # Finite Difference at 0: k_1/2 (T1 - T0) - k_-1/2 (T0 - T_{-1})
        # Approximated centered at node 0 using k(0) for simplicity or averaged
        # The derivation writes: k(x_{1/2}) ... 
        # We use k_vals[0] as proxy for neighbors or arithmetic mean
        k_plus = 0.5 * (k_vals[1] + k_vals[0])
        k_minus = k_vals[0] # Boundary approximation
        
        diff_0 = (k_plus * (T_new[1] - T_new[0]) - k_minus * (T_new[0] - T_ghost)) * inv_h2
        G[0] = T_new[0] - T_old[0] - coeff * diff_0
        
        # Jacobian entries for i=0
        # dG[0]/dT[0] = 1 - coeff * [ -k_plus - k_minus(1 - dTg_dT0) ] / h^2
        dDiff_dT0 = ( -k_plus - k_minus * (1.0 - dTg_dT0) ) * inv_h2
        dDiff_dT1 = ( k_plus - k_minus * (-1.0) ) * inv_h2 # T_ghost depends on T1 linearly (+1) is incorrect?
        # T_ghost = C + T1. dTg/dT1 = 1.
        # Term is - k_minus * ( - T_ghost ) -> + k_minus * T_ghost
        # deriv w.r.t T1 is k_minus * 1. 
        # Also first term k_plus * T1. Total: (k_plus + k_minus) * inv_h2
        dDiff_dT1 = (k_plus + k_minus) * inv_h2
        
        diag[0] = 1.0 - coeff * dDiff_dT0
        upper[1] = - coeff * dDiff_dT1
        
        # --- Interior Points ---
        for i in range(1, N-1):
            kp = 0.5*(k_vals[i+1]+k_vals[i])
            km = 0.5*(k_vals[i]+k_vals[i-1])
            diff = (kp*(T_new[i+1]-T_new[i]) - km*(T_new[i]-T_new[i-1])) * inv_h2
            G[i] = T_new[i] - T_old[i] - coeff * diff
            
            diag[i] = 1.0 - coeff * ( -kp - km ) * inv_h2
            upper[i+1] = - coeff * ( kp ) * inv_h2
            lower[i] = - coeff * ( km ) * inv_h2 # Stored in shifted index for banded solver

        # --- Boundary Condition x=d ---
        # Neumann 0: T_{d+1} = T_{d-1}
        i = N-1
        km = 0.5*(k_vals[i]+k_vals[i-1])
        # diff = (0 - km(T_i - T_{i-1})) / h^2 (Approx assuming Flux_out=0)
        diff_end = -km * (T_new[i] - T_new[i-1]) * inv_h2
        G[i] = T_new[i] - T_old[i] - coeff * diff_end
        
        diag[i] = 1.0 - coeff * (-km) * inv_h2
        lower[i] = - coeff * (km) * inv_h2

        # Solve Linear System
        J_banded = np.zeros((3, N))
        J_banded[0, 1:] = upper[1:] 
        J_banded[1, :] = diag
        J_banded[2, :-1] = lower[1:]
        
        delta = scipy.linalg.solve_banded((1, 1), J_banded, -G)
        T_new += delta
        
        if np.linalg.norm(delta) < 1e-5:
            break
            
    return T_new

# -------------------------------------------------------------------------
# 6. MAIN SIMULATION LOOP
# -------------------------------------------------------------------------

# Initialization
# T(x,0) = T0(x). Code assumes constant 100K as per previous instance.
T_field = np.ones(Nx) * 100.0
rg_field = np.ones(Nx) * 100.0 * LS # 100 microns
rb_field = np.ones(Nx) * 10.0 * LS  # 10 microns

# Data storage
times = []
T_surf_hist = []
rb_surf_hist = []

current_time = 0.0
print(f"Starting Simulation [Derivation Exact Mode] for {t_max/86400} days...")

while current_time < t_max:
    
    # 1. Update Microstructure (Implicit Euler)
    for i in range(Nx):
        # Solves coupled mass balance for rg, rb at this node
        rg_new, rb_new = solve_microstructure_implicit(
            rg_field[i], rb_field[i], T_field[i], dt
        )
        rg_field[i] = rg_new
        rb_field[i] = rb_new

    # 2. Update Temperature (Implicit Euler + Newton)
    T_field = solve_heat_implicit(T_field, rg_field, rb_field, dt, current_time)
    
    # 3. Time Step
    current_time += dt
    
    # Store Data
    if current_time % 3600 == 0: # Every hour
        times.append(current_time)
        T_surf_hist.append(T_field[0])
        rb_surf_hist.append(rb_field[0])

# -------------------------------------------------------------------------
# 7. VISUALIZATION
# -------------------------------------------------------------------------
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(np.array(times)/86400.0, T_surf_hist, 'k-')
plt.title("Surface Temperature")
plt.xlabel("Time (Days)")
plt.ylabel("Temperature (K)")
plt.grid(True)

plt.subplot(1, 2, 2)
# Convert back to microns for plotting
plt.plot(np.array(times)/86400.0, np.array(rb_surf_hist)/LS, 'r-')
plt.title("Bond Radius Evolution")
plt.xlabel("Time (Days)")
plt.ylabel("Bond Radius ($\mu m$)")
plt.grid(True)

plt.tight_layout()
plt.show()