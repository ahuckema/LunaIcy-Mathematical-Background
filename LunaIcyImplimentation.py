import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import root, brentq

# --- 1. Constants & Configuration ---
# Physical Constants (SI)
R = 8.314           # J/(mol K)
M = 0.018015        # kg/mol (Molar mass of water)
rho0 = 917.0        # kg/m^3 (Ice density)
gamma = 0.065       # J/m^2 (Surface tension)
P0 = 3.5e12         # Pa (Pre-exponential factor)
Q_sub = 51000.0     # J/mol (Activation energy)
sigma_SB = 5.67e-8  # Stefan-Boltzmann

# Simulation Parameters
alpha = 0.03        # Sticking coefficient
phi = 0.4           # Porosity
T_celsius = -3.0
T_val = 273.15 + T_celsius

# Scaling (Microns) to keep numerics stable
LS = 1e-6           # Length Scale (meters)
AS = LS**2          # Area Scale
VS = LS**3          # Volume Scale

# --- 2. Geometry Module (The Derivations) ---

def get_geometry(r_g_um, r_b_um):
    """
    Calculates Volume, Area, and Curvature based on the LaTeX derivation.
    Input: radii in Microns.
    Output: SI Units (m^3, m^2, 1/m).
    """
    # 1. Derived Radius rp (Fillet)
    # Formula: rp = rb^2 / (2*(rg - rb))
    if r_b_um >= r_g_um * 0.999: r_b_um = r_g_um * 0.999 # Stability clamp
    r_p_um = (r_b_um**2) / (2 * (r_g_um - r_b_um))
    
    # 2. Intersection x_star (Height where grain meets bond)
    # Geometric constraint: h_grain = h_bond
    # sqrt(rg^2 - x^2) = (rb + rp) - sqrt(rp^2 - (rg - x)^2)
    def h_diff(x):
        hg = np.sqrt(max(0, r_g_um**2 - x**2))
        hb = (r_b_um + r_p_um) - np.sqrt(max(0, r_p_um**2 - (r_g_um - x)**2))
        return hg - hb
    
    try:
        x_star_um = brentq(h_diff, 0, r_g_um * 0.999)
    except:
        x_star_um = r_g_um * 0.9 # Fallback if geometry is degenerate

    # --- Grain Integrals (Sections 5.1 & 5.3) ---
    
    # The integral A(x*, r) from text
    def integrand_A(theta, r):
        # sqrt(1 - (x*/(r*cos(theta)))^2)
        val = 1 - (x_star_um / (r * np.cos(theta)))**2
        return np.sqrt(max(0, val))

    # Calculate A(x*, rg) for Surface Area
    limit_S = np.arccos(min(1.0, x_star_um / r_g_um))
    A_val_S, _ = quad(integrand_A, 0, limit_S, args=(r_g_um,))
    
    # S_g Formula (Text calculates 1/4th, we multiply by 4)
    # S_g = 4 * rg^2 * (pi - 2 * A)
    S_g_um2 = 4 * (r_g_um**2 * (np.pi - 2 * A_val_S))

    # Calculate V_g (Text formula involving integral over r)
    # V_g = 4 * [ pi*x*^3/3 + pi(rg^3/3 - x*^3/3) - 2 * int(r^2 * A(x*,r)) ]
    
    def rad_integrand(r):
        lim = np.arccos(min(1.0, x_star_um / r))
        # Nested quad for A(x*, r)
        a_val, _ = quad(integrand_A, 0, lim, args=(r,))
        return (r**2) * a_val

    # Integral part
    int_radial, _ = quad(rad_integrand, x_star_um, r_g_um)
    
    term1 = (np.pi * x_star_um**3) / 3.0
    term2 = np.pi * ((r_g_um**3 / 3.0) - (x_star_um**3 / 3.0))
    V_g_um3 = 4 * (term1 + term2 - 2 * int_radial)

    # --- Bond Integrals (Sections 5.2 & 5.3) ---
    # Note: Text formula derives volume for one side of the neck.
    # We assume 2 bonds per grain (1D chain), so we multiply V_b and S_b by 2.
    
    L_um = r_g_um - x_star_um # Integration limit
    
    # S_b (Text 5.3)
    # S_b_single = 2*pi*rp * [ (rb+rp)*arcsin(...) - L ]
    term_asin = np.arcsin(min(1.0, L_um / r_p_um))
    S_b_single = 2 * np.pi * r_p_um * ((r_b_um + r_p_um) * term_asin - L_um)
    S_b_um2 = 2 * S_b_single # Two bonds
    
    # V_b (Text 5.2)
    # A = pi * [ (L/2)*sqrt(...) + (rp^2/2)*arcsin(...) ]
    term_sqrt = np.sqrt(max(0, r_p_um**2 - L_um**2))
    A_term = np.pi * ((L_um / 2) * term_sqrt + (r_p_um**2 / 2) * term_asin)
    
    # V_single = pi(rb+rp)^2 * L - 2(rb+rp)A + pi*rp^2*L - (pi/3)L^3
    t1 = np.pi * (r_b_um + r_p_um)**2 * L_um
    t2 = 2 * (r_b_um + r_p_um) * A_term
    t3 = np.pi * r_p_um**2 * L_um
    t4 = (np.pi / 3) * L_um**3
    
    V_b_single = t1 - t2 + t3 - t4
    V_b_um3 = 2 * V_b_single # Two bonds

    # --- Curvatures ---
    # Text: Kg = 2/rg. 
    # Text: Kb = 2/rb (This is written in text, but physics requires concave/negative)
    # We use Kb = -1/rp (Concave neck curvature) to drive sintering.
    Kg = 2.0 / (r_g_um * LS)
    Kb = -1.0 / (r_p_um * LS) 

    return (V_g_um3 * VS, V_b_um3 * VS, 
            S_g_um2 * AS, S_b_um2 * AS, 
            Kg, Kb)

# --- 3. Physics & Solver ---

def get_mass_fluxes(state, T):
    """
    Calculates m_gas and fluxes Jg, Jb based on Section 2 & 3.
    """
    rg, rb = state
    Vg, Vb, Sg, Sb, Kg, Kb = get_geometry(rg, rb)
    
    # Saturation Pressure (Flat)
    P_sat = P0 * np.exp(-Q_sub / (R * T))
    
    # Kelvin Pressures (Section 2.1)
    # P_Kj = P_sat * (1 + (gamma * M / (R * T * rho0)) * Kj)
    kelvin_factor = (gamma * M) / (R * T * rho0)
    P_Kg = P_sat * (1 + kelvin_factor * Kg)
    P_Kb = P_sat * (1 + kelvin_factor * Kb)
    
    # Equilibrium Gas Mass (Section 3.1 Formula)
    # m_gas = P_sat * (Stot + Kelvin_term) * (M Vg phi) / ((1-phi) R T Stot)
    S_tot = Sg + Sb
    kelvin_sum = Sg * Kg + Sb * Kb
    
    numerator = P_sat * (S_tot + kelvin_factor * kelvin_sum)
    geometric_factor = (M * Vg * phi) / ((1 - phi) * R * T * S_tot)
    
    m_gas = numerator * geometric_factor
    
    # Gas Pressure (Section 2.1)
    # P_gas = (1-phi) * m_gas * R * T / (M * Vg * phi)
    P_gas = ((1 - phi) * m_gas * R * T) / (M * Vg * phi)
    
    # Fluxes (Hertz-Knudsen)
    # J = alpha * (P_surface - P_gas) * sqrt(M / 2 pi R T)
    kinetics = alpha * np.sqrt(M / (2 * np.pi * R * T))
    
    J_g = kinetics * (P_Kg - P_gas)
    J_b = kinetics * (P_Kb - P_gas)
    
    return J_g, J_b, Sg, Sb, Vg, Vb

def inverse_geometry(target_Vg, target_Vb, guess_r):
    """
    Solves r_g, r_b given V_g, V_b.
    """
    def resid(r):
        # Input r in microns, output residual in SI volume
        # Clamp to avoid physical violation
        if r[1] >= r[0]: return [1e-15, 1e-15] 
        if r[1] < 1e-4: return [1e-15, 1e-15]
        
        v_g, v_b, _, _, _, _ = get_geometry(r[0], r[1])
        return [v_g - target_Vg, v_b - target_Vb]

    # Use 'hybr' or 'lm'. 
    sol = root(resid, guess_r, method='lm', tol=1e-15)
    if sol.success:
        return sol.x
    else:
        return guess_r # Fallback: no change

# --- 4. Main Simulation Loop ---

def run_simulation(r0_um, t_max):
    # Initial State
    rg_um = r0_um
    rb_um = r0_um * 0.1 # Start with small neck
    
    current_time = 0
    dt = 1e-2 # Start small
    
    # History
    times = []
    ratios = []
    
    next_record = 0
    
    print(f"Start: rg={rg_um}um, T={T_celsius}C")
    
    while current_time < t_max:
        
        # 1. Calculate Geometry & Physics at current step t
        J_g, J_b, Sg, Sb, Vg_old, Vb_old = get_mass_fluxes([rg_um, rb_um], T_val)
        
        # 2. Update Mass (Section 3.2: dmi/dt = - Ji * Si)
        # "Assume constant grain/bond geometry in a single time step"
        m_g_old = Vg_old * rho0
        m_b_old = Vb_old * rho0
        
        m_g_new = m_g_old - J_g * Sg * dt
        m_b_new = m_b_old - J_b * Sb * dt
        
        # 3. Retrieve Volumes
        V_g_new = m_g_new / rho0
        V_b_new = m_b_new / rho0
        
        # 4. Inverse Problem: Get radii from new volumes
        new_radii = inverse_geometry(V_g_new, V_b_new, [rg_um, rb_um])
        rg_um, rb_um = new_radii
        
        # Recording & Time stepping
        current_time += dt
        
        # Adaptive time step (simple heuristic based on rate of change)
        ratio = rb_um / rg_um
        if current_time >= next_record:
            times.append(current_time)
            ratios.append(ratio)
            next_record = max(dt, current_time * 1.1)
            
            # Accelerate time step as system relaxes
            dt *= 1.1
        
        if ratio > 0.6: break # Sintering limit
        
    return np.array(times), np.array(ratios)

# --- 5. Visualization ---

plt.figure(figsize=(10, 6))
grain_sizes = [40, 100, 220] # Microns
colors = ['blue', 'orange', 'green']

for r0, col in zip(grain_sizes, colors):
    t, x = run_simulation(r0, 5e6)
    plt.loglog(t, x, label=f'$r_g = {r0} \mu m$', color=col, linewidth=2)

plt.xlabel("Time (s)")
plt.ylabel("Neck Ratio $x = r_b / r_g$")
plt.title(f"Reimplemented LunaIcy Sintering\n$T={T_celsius}^\circ C, \phi={phi}, \\alpha={alpha}$")
plt.grid(True, which="both", alpha=0.3)
plt.legend()
plt.show()