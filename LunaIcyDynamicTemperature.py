import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import root, brentq
import scipy.linalg

# =========================================================================
# 1. PHYSICAL CONSTANTS & CONFIGURATION
# =========================================================================

# --- SI Units ---
R_gas = 8.314        # J/(mol K)
M_h2o = 0.018015     # kg/mol
rho0 = 917.0         # kg/m^3 (Bulk Ice density)
gamma = 0.065        # J/m^2 (Surface tension)
P0 = 3.5e12          # Pa (Vapor pressure constant)
Q_sub = 51000.0      # J/mol (Sublimation energy)
sigma_SB = 5.67e-8   # Stefan-Boltzmann constant
cp_ice = 2100.0      # J/(kg K) - Heat capacity
epsilon = 0.9        # Emissivity

# --- Simulation Parameters ---
alpha = 0.03         # Sticking coefficient
phi = 0.4            # Porosity
# We can now take larger time steps thanks to Implicit Euler
dt = 3600.0          # Time step (s) -> 1 Hour steps
t_max = 50 * 86400   # Simulate 50 Earth days

# --- Grid Setup (1D Slice) ---
depth = 0.2          # meters (20 cm)
Nx = 30              # Number of spatial nodes
dx = depth / Nx
x_grid = np.linspace(dx/2, depth - dx/2, Nx) # Cell-centered grid

# --- Solar Forcing ---
day_length = 3.55 * 86400 # Europa orbital period (seconds)
solar_flux_max = 50.0     # W/m^2 (Mean Solar Flux at Jupiter)

# --- Microscale Scaling (Microns) ---
# Used to keep numerical values for radii O(1) instead of O(1e-6)
LS = 1e-6 
AS = LS**2 
VS = LS**3 

# =========================================================================
# 2. MICROSTRUCTURE GEOMETRY & PHYSICS (ODEs)
# =========================================================================

def get_geometry(r_g_um, r_b_um):
    """
    Calculates Volume (m3), Area (m2), and Curvature (1/m) 
    based on the overlapping sphere geometry derived in the text.
    """
    # Stability: bond cannot exceed grain radius
    if r_b_um >= r_g_um * 0.999: r_b_um = r_g_um * 0.999 
    
    # 1. Derived Radius rp (The fillet radius)
    r_p_um = (r_b_um**2) / (2 * (r_g_um - r_b_um))
    
    # 2. Intersection x_star (Height where grain meets bond neck)
    def h_diff(x):
        hg = np.sqrt(max(0, r_g_um**2 - x**2))
        hb = (r_b_um + r_p_um) - np.sqrt(max(0, r_p_um**2 - (r_g_um - x)**2))
        return hg - hb
    
    try:
        x_star_um = brentq(h_diff, 0, r_g_um * 0.999)
    except:
        x_star_um = r_g_um * 0.9 # Fallback

    # --- Grain Geometry Integrals ---
    def integrand_A(theta, r):
        val = 1 - (x_star_um / (r * np.cos(theta)))**2
        return np.sqrt(max(0, val))

    limit_S = np.arccos(min(1.0, x_star_um / r_g_um))
    A_val_S, _ = quad(integrand_A, 0, limit_S, args=(r_g_um,))
    S_g_um2 = 4 * (r_g_um**2 * (np.pi - 2 * A_val_S))

    def rad_integrand(r):
        lim = np.arccos(min(1.0, x_star_um / r))
        a_val, _ = quad(integrand_A, 0, lim, args=(r,))
        return (r**2) * a_val

    int_radial, _ = quad(rad_integrand, x_star_um, r_g_um)
    term1 = (np.pi * x_star_um**3) / 3.0
    term2 = np.pi * ((r_g_um**3 / 3.0) - (x_star_um**3 / 3.0))
    V_g_um3 = 4 * (term1 + term2 - 2 * int_radial)

    # --- Bond Geometry Integrals ---
    L_um = r_g_um - x_star_um
    term_asin = np.arcsin(min(1.0, L_um / r_p_um))
    S_b_single = 2 * np.pi * r_p_um * ((r_b_um + r_p_um) * term_asin - L_um)
    S_b_um2 = 2 * S_b_single 
    
    term_sqrt = np.sqrt(max(0, r_p_um**2 - L_um**2))
    A_term = np.pi * ((L_um / 2) * term_sqrt + (r_p_um**2 / 2) * term_asin)
    t1 = np.pi * (r_b_um + r_p_um)**2 * L_um
    t2 = 2 * (r_b_um + r_p_um) * A_term
    t3 = np.pi * r_p_um**2 * L_um
    t4 = (np.pi / 3) * L_um**3
    V_b_single = t1 - t2 + t3 - t4
    V_b_um3 = 2 * V_b_single

    # --- Curvatures ---
    Kg = 2.0 / (r_g_um * LS)
    Kb = -1.0 / (r_p_um * LS) # Negative curvature drives sintering

    return (V_g_um3 * VS, V_b_um3 * VS, S_g_um2 * AS, S_b_um2 * AS, Kg, Kb)

def get_mass_fluxes(rg, rb, T):
    """Calculates Vapor Fluxes (kg/s/m2) using Hertz-Knudsen."""
    Vg, Vb, Sg, Sb, Kg, Kb = get_geometry(rg, rb)
    
    P_sat = P0 * np.exp(-Q_sub / (R_gas * T))
    
    # Kelvin Equation Corrections
    kelvin_factor = (gamma * M_h2o) / (R_gas * T * rho0)
    P_Kg = P_sat * (1 + kelvin_factor * Kg)
    P_Kb = P_sat * (1 + kelvin_factor * Kb)
    
    # Equilibrium Pore Pressure (Conservation of Mass)
    S_tot = Sg + Sb
    kelvin_sum = Sg * Kg + Sb * Kb
    
    numerator = P_sat * (S_tot + kelvin_factor * kelvin_sum)
    geometric_factor = (M_h2o * Vg * phi) / ((1 - phi) * R_gas * T * S_tot)
    
    m_gas = numerator * geometric_factor
    P_gas = ((1 - phi) * m_gas * R_gas * T) / (M_h2o * Vg * phi)
    
    # Fluxes
    kinetics = alpha * np.sqrt(M_h2o / (2 * np.pi * R_gas * T))
    J_g = kinetics * (P_Kg - P_gas)
    J_b = kinetics * (P_Kb - P_gas)
    
    return J_g, J_b, Sg, Sb, Vg, Vb

def inverse_geometry(target_Vg, target_Vb, guess_r):
    """Recovers radii from new masses/volumes."""
    def resid(r):
        if r[1] >= r[0] or r[1] < 1e-4: return [1e9, 1e9]
        v_g, v_b, _, _, _, _ = get_geometry(r[0], r[1])
        return [v_g - target_Vg, v_b - target_Vb]

    # Levenberg-Marquardt is robust for this
    sol = root(resid, guess_r, method='lm', tol=1e-18)
    return sol.x if sol.success else guess_r

# =========================================================================
# 3. HEAT EQUATION HELPERS
# =========================================================================

def calc_solar_flux(t):
    """Simple Day/Night cosine model."""
    theta_i = (2 * np.pi * t) / day_length
    cos_theta = np.cos(theta_i)
    return max(0.0, solar_flux_max * cos_theta)

def calc_conductivity(T_arr, rg_arr, rb_arr):
    """
    Thermal Conductivity k(T, geometry).
    Coupling: k depends on bond-to-grain ratio.
    """
    k0 = 567.0 / T_arr  # Klinger (1980) for pure ice
    # The sintering neck acts as a throttle for heat flow
    geom_factor = rb_arr / rg_arr
    k_eff = k0 * (1 - phi) * geom_factor
    return k_eff

# =========================================================================
# 4. IMPLICIT SOLVER (Newton-Raphson)
# =========================================================================

def residual_and_jacobian(T_guess, T_old, rg, rb, dt, dx, time_curr):
    """
    Computes G(T) = 0 and Jacobian matrix J for Implicit Euler.
    Includes Non-linear Radiation Boundary Condition.
    """
    N = len(T_guess)
    G = np.zeros(N)
    
    # Pre-compute material properties
    k_vals = calc_conductivity(T_guess, rg, rb)
    rho_eff = rho0 * (1 - phi)
    cap = rho_eff * cp_ice
    inv_dx2 = 1.0 / (dx**2)
    coeff = dt / cap
    
    # Jacobian Banded Storage (3 rows x N cols)
    # Row 0: Upper diag (i, i+1)
    # Row 1: Main diag (i, i)
    # Row 2: Lower diag (i, i-1)
    diag = np.zeros(N)
    upper = np.zeros(N) # Shifted: upper[i] corresponds to J[i, i+1]
    lower = np.zeros(N) # Shifted: lower[i] corresponds to J[i, i-1]

    # --- TOP BOUNDARY (i=0) ---
    F_sol = calc_solar_flux(time_curr)
    Rad = epsilon * sigma_SB * T_guess[0]**4
    dRad_dT = 4 * epsilon * sigma_SB * T_guess[0]**3
    
    # Ghost Point reconstruction: T_{-1}
    # k * (T_0 - T_{-1})/dx = -F + Rad  => T_{-1} = T_0 - (dx/k)*(-F + Rad)
    T_ghost = T_guess[0] + (2*dx/k_vals[0]) * (F_sol - Rad)
    dTg_dT0 = 1.0 - (2*dx/k_vals[0]) * dRad_dT
    
    # Diffusion term at 0
    diff_0 = k_vals[0] * (T_guess[1] - T_guess[0]) * inv_dx2 - \
             k_vals[0] * (T_guess[0] - T_ghost) * inv_dx2
             
    G[0] = T_guess[0] - T_old[0] - coeff * diff_0
    
    # Jacobian 0
    dD0_dT0 = k_vals[0] * inv_dx2 * (-1.0 - (1.0 - dTg_dT0)) # -1 from T0, -1 from -T_ghost
    # Actually: d/dT0 [ k(T1 - T0) - k(T0 - Tg) ] = k(-1) - k(1 - dTg/dT0) = k(-2 + dTg/dT0)
    dD0_dT0 = k_vals[0] * inv_dx2 * (-2.0 + dTg_dT0)
    dD0_dT1 = k_vals[0] * inv_dx2 * (1.0)
    
    diag[0] = 1.0 - coeff * dD0_dT0
    upper[1] = - coeff * dD0_dT1 # Stored at index 1 for solve_banded logic (col 1)

    # --- INTERIOR (i=1 to N-2) ---
    for i in range(1, N-1):
        k_plus = 0.5 * (k_vals[i+1] + k_vals[i])
        k_minus = 0.5 * (k_vals[i] + k_vals[i-1])
        
        diff = (k_plus * (T_guess[i+1] - T_guess[i]) - 
                k_minus * (T_guess[i] - T_guess[i-1])) * inv_dx2
        
        G[i] = T_guess[i] - T_old[i] - coeff * diff
        
        dD_dTi   = -(k_plus + k_minus) * inv_dx2
        dD_dTkm1 = k_minus * inv_dx2
        dD_dTkp1 = k_plus * inv_dx2
        
        diag[i] = 1.0 - coeff * dD_dTi
        lower[i] = - coeff * dD_dTkm1 # J[i, i-1]
        upper[i+1] = - coeff * dD_dTkp1 # J[i, i+1]

    # --- BOTTOM BOUNDARY (i=N-1) ---
    # Neumann BC: T_{d+1} = T_{d-1}. 
    # Finite Diff at N-1 becomes: k+ (T_{N-2} - T_{N-1}) - k- (T_{N-1} - T_{N-2})
    # Essentially flux from right is 0.
    i = N - 1
    k_minus = 0.5 * (k_vals[i] + k_vals[i-1])
    
    # We treat the boundary as adiabatic: No flux out the bottom.
    # Flux_in = k_minus * (T_{N-2} - T_{N-1}) / dx
    # Flux_out = 0
    # Div = (Flux_out - Flux_in) / dx = -Flux_in / dx
    diff_last = - (k_minus * (T_guess[i] - T_guess[i-1])) * inv_dx2
    
    G[i] = T_guess[i] - T_old[i] - coeff * diff_last
    
    dD_dTi = -k_minus * inv_dx2
    dD_dTkm1 = k_minus * inv_dx2
    
    diag[i] = 1.0 - coeff * dD_dTi
    lower[i] = - coeff * dD_dTkm1

    # Pack for scipy: 
    # Row 0: Upper diagonal (element 0 is ignored)
    # Row 1: Main diagonal
    # Row 2: Lower diagonal (element N-1 is ignored)
    J_banded = np.zeros((3, N))
    J_banded[0, :] = upper 
    J_banded[1, :] = diag
    J_banded[2, :] = lower
    
    return G, J_banded

def step_implicit_euler(T_current, rg, rb, dt, dx, time_curr):
    """Solves for T_new using Newton-Raphson."""
    T_new = T_current.copy()
    tol = 1e-5
    max_iter = 15
    
    for k in range(max_iter):
        G, J_banded = residual_and_jacobian(T_new, T_current, rg, rb, dt, dx, time_curr)
        
        if np.linalg.norm(G, np.inf) < tol:
            return T_new
            
        # Solve J * delta = -G
        # Use specialized O(N) solver for banded matrices
        try:
            delta = scipy.linalg.solve_banded((1, 1), J_banded, -G)
            T_new += delta
        except np.linalg.LinAlgError:
            print("Matrix singular, breaking.")
            break
            
    return T_new

# =========================================================================
# 5. MAIN SIMULATION LOOP
# =========================================================================

# Initial Conditions
T_field = np.ones(Nx) * 100.0   # Start cold (100 K)
rg_field = np.ones(Nx) * 100.0  # 100 microns
rb_field = np.ones(Nx) * 10.0   # 10 microns (small necks)

# Storage for Plotting
rb_initial = rb_field.copy()
T_history_surf = []
time_points = []

# --- Acceleration Factor ---
# REALITY CHECK: Sintering takes millions of years. 
# We simulate days here. To see ANY change in Figure 8, 
# we artificially boost the mass flux. 
# Set this to 1.0 for a "Scientific" run (which would need t_max = 1e6 years)
ACCELERATION = 20000.0 

print(f"Starting Implicit Simulation for {t_max/86400:.1f} days...")
current_time = 0.0

while current_time < t_max:
    
    # 1. Update Microstructure (ODE Step)
    # We iterate over every spatial node
    for i in range(Nx):
        J_g, J_b, Sg, Sb, Vg_old, Vb_old = get_mass_fluxes(rg_field[i], rb_field[i], T_field[i])
        
        # Explicit update for mass (Still valid as dt is small enough for mass dynamics usually)
        # Using the ACCELERATION factor for visualization
        m_g_new = (Vg_old * rho0) - J_g * Sg * dt * ACCELERATION
        m_b_new = (Vb_old * rho0) - J_b * Sb * dt * ACCELERATION
        
        # Recover Radii
        new_radii = inverse_geometry(m_g_new/rho0, m_b_new/rho0, [rg_field[i], rb_field[i]])
        rg_field[i], rb_field[i] = new_radii

    # 2. Update Temperature (PDE Step - Implicit)
    T_field = step_implicit_euler(T_field, rg_field, rb_field, dt, dx, current_time)
    
    # 3. Time Increment
    current_time += dt
    
    # Logging
    if current_time % (day_length/4) < dt:
        time_points.append(current_time)
        T_history_surf.append(T_field[0])
        print(f"Day {current_time/86400:.2f} | T_surf: {T_field[0]:.2f} K | rb_surf: {rb_field[0]:.4f}")

# =========================================================================
# 6. VISUALIZATION (Figure 8 Style)
# =========================================================================


plt.figure(figsize=(12, 5))

# Subplot 1: Bond Radius Profile (The Figure 8 Request)
plt.subplot(1, 2, 1)
plt.plot(x_grid * 100, rb_initial, 'b--', label='Initial $t=0$')
plt.plot(x_grid * 100, rb_field, 'r-o', label=f'Final $t={t_max/86400:.0f}$d')
plt.xlabel("Depth (cm)")
plt.ylabel("Bond Radius $r_b$ ($\mu m$)")
plt.title(f"Sintering Profile\n(Accel Factor x{ACCELERATION:.0e})")
plt.legend()
plt.grid(True, alpha=0.3)

# Subplot 2: Thermal History
plt.subplot(1, 2, 2)
plt.plot(np.array(time_points)/86400, T_history_surf, 'k-')
plt.xlabel("Time (Days)")
plt.ylabel("Surface Temperature (K)")
plt.title("Surface Temperature Response")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()