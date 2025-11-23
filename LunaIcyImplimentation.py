import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root
from scipy.integrate import quad

# ==========================================
# 1. PHYSICAL CONSTANTS & PARAMETERS
# ==========================================
R = 8.314          # Universal gas constant [J/(mol K)]
M = 0.018015       # Molar mass of water [kg/mol]
rho0 = 917.0       # Density of bulk ice [kg/m^3]
gamma = 0.065      # Surface tension of ice [N/m]
P0 = 3.5e12        # Pre-exponential factor [Pa]
Q_sub = 51000.0    # Sublimation energy [J/mol]

# Simulation specific
alpha = 0.03       # Sticking coefficient
T_celsius = -3.0
T = 273.15 + T_celsius
phi = 0.3          # Porosity (used for m_gas calculation)

# NUMERICAL SCALING FACTORS
# We compute geometry in microns to avoid machine epsilon issues with 1e-18 numbers
L_SCALE = 1e-6     
V_SCALE = L_SCALE**3
A_SCALE = L_SCALE**2

# ==========================================
# 2. GEOMETRY FUNCTIONS (The LunaIcy Model)
# ==========================================

def get_rp(rg, rb):
    """Calculates pore/fillet radius based on user formula."""
    # rg, rb in MICRONS
    if rg <= rb: return 1e-9
    # Formula: rp = rb^2 / (2*(rg - rb))
    return (rb**2) / (2 * (rg - rb))

def get_geometry_scaled(vars_micron):
    """
    Computes Volumes (m^3), Areas (m^2), and rp (m) 
    Input: [rg, rb] in MICRONS
    """
    rg_um, rb_um = vars_micron
    
    # 1. Geometric Safety & Constraints
    if rb_um <= 1e-9: rb_um = 1e-4 # Avoid div by zero
    if rb_um >= rg_um * 0.99: rb_um = rg_um * 0.99
    
    rp_um = get_rp(rg_um, rb_um)
    
    # 2. Find Intersection x* (height where grain meets bond)
    # Using the rigorous geometric definition: 
    # Height_Grain(x*) = Height_Bond(x*)
    def h_diff(x):
        # Height of Grain sphere
        hg = np.sqrt(max(0, rg_um**2 - x**2))
        # Height of Bond torus/fillet
        # The fillet is a circle of radius rp centered at (rg, rb+rp)
        # Equation: (x - rg)^2 + (y - (rb+rp))^2 = rp^2
        # y = (rb+rp) - sqrt(rp^2 - (rg-x)^2)
        hb = (rb_um + rp_um) - np.sqrt(max(0, rp_um**2 - (rg_um - x)**2))
        return hg - hb

    # Bracket search for x* between 0 and rg
    try:
        # Standard brentq is fast and robust
        from scipy.optimize import brentq
        x_star_um = brentq(h_diff, 0, rg_um * 0.9999)
    except:
        # Fallback if geometry is degenerate (very rare)
        x_star_um = rg_um * 0.9 

    # 3. Surface Areas (Integrals)
    # S_g: Grain Area
    def integrand_Sg(theta):
        val = 1 - (x_star_um / (rg_um * np.cos(theta)))**2
        return np.sqrt(max(0, val))
    
    limit = np.arccos(min(1.0, x_star_um/rg_um))
    val_Sg, _ = quad(integrand_Sg, 0, limit)
    # Multiplied by 4 per user note/symmetry
    Sg_um2 = 4 * (rg_um**2 * (np.pi - 2 * val_Sg))
    
    # S_b: Bond Area (Explicit formula)
    L_um = rg_um - x_star_um
    term_asin = np.arcsin(min(1.0, L_um/rp_um))
    Sb_um2 = 2 * np.pi * rp_um * ((rb_um + rp_um)*term_asin - L_um)
    
    # 4. Volumes (Integrals)
    # V_g: Grain Volume
    def A_func(r): 
        # The inner integral A(x*, r) from the notes
        if r < x_star_um: return 0
        up = np.arccos(min(1.0, x_star_um/r))
        def sub(th): return np.sqrt(max(0, 1-(x_star_um/(r*np.cos(th)))**2))
        v, _ = quad(sub, 0, up)
        return v
    
    def rad_int(r):
        return r**2 * (np.pi - 2*A_func(r))
    
    V_outer, _ = quad(rad_int, x_star_um, rg_um)
    V_inner = (np.pi * x_star_um**3)/3.0
    Vg_um3 = 4 * (V_inner + V_outer)
    
    # V_b: Bond Volume (Explicit formula)
    t1 = np.pi * (rb_um + rp_um)**2 * L_um
    sq_term = np.sqrt(max(0, rp_um**2 - L_um**2))
    t2 = 2*(rb_um + rp_um)*np.pi*((L_um/2)*sq_term + (rp_um**2/2)*term_asin)
    t3 = np.pi * rp_um**2 * L_um
    t4 = (np.pi/3) * L_um**3
    Vb_um3 = t1 - t2 + t3 - t4
    
    # 5. Return in SI Units (Meters)
    return (Vg_um3 * V_SCALE, Vb_um3 * V_SCALE, 
            Sg_um2 * A_SCALE, Sb_um2 * A_SCALE, 
            rp_um * L_SCALE)

# ==========================================
# 3. SIMULATION LOOP (Method of Lines)
# ==========================================

def run_simulation(r0_meters):
    # --- Initialization ---
    rg_um = r0_meters / L_SCALE
    rb_um = rg_um * 0.15 # Start with a small 15% neck
    
    # Physics constants pre-calculation
    P_sat = P0 * np.exp(-Q_sub / (R * T))
    Kelvin_Base = (gamma * M) / (R * T * rho0)
    Flux_Prefactor = alpha * np.sqrt(M / (2 * np.pi * R * T))
    
    # Time Stepping Strategy
    # Large grains sinter strictly slower. We adapt dt and t_max based on size.
    if r0_meters < 5e-6:      # < 5 micron
        dt = 0.5; t_max = 5000
    elif r0_meters < 50e-6:   # < 50 micron
        dt = 50.0; t_max = 500000 
    else:                     # Large grains
        dt = 500.0; t_max = 5e6 
        
    t_vals = []
    ratio_vals = []
    current_time = 0
    
    # Initial Mass
    Vg, Vb, Sg, Sb, rp = get_geometry_scaled([rg_um, rb_um])
    m_g = Vg * rho0
    m_b = Vb * rho0
    
    print(f"Simulating r0={rg_um}um | T={T:.1f}K | P_sat={P_sat:.2e} Pa")

    while current_time < t_max:
        # --- A. Physics Update ---
        
        # 1. Curvatures
        Kg = 2.0 / (rg_um * L_SCALE)
        Kb = -1.0 / rp  # Concave neck (Negative Curvature)
        
        # 2. Surface Pressures (EXPONENTIAL FORM)
        # Using exp ensures P > 0 and prevents the "inverted" artifact
        P_Kg = P_sat * np.exp(Kelvin_Base * Kg)
        P_Kb = P_sat * np.exp(Kelvin_Base * Kb)
        
        # 3. Equilibrium Gas Pressure
        # Derived from dm_gas/dt = 0 => Sum(Flux*Area) = 0
        # P_gas = (P_Kg*Sg + P_Kb*Sb) / (Sg + Sb)
        P_gas = (P_Kg * Sg + P_Kb * Sb) / (Sg + Sb)
        
        # (Optional) Recover m_gas for the record, though not used for flux
        m_gas = (P_gas * M * Vg * phi) / ((1 - phi) * R * T)
        
        # 4. Fluxes (Hertz-Knudsen)
        J_g = Flux_Prefactor * (P_Kg - P_gas) # > 0 (Sublimation)
        J_b = Flux_Prefactor * (P_Kb - P_gas) # < 0 (Deposition)
        
        # --- B. Mass Update (Explicit Euler) ---
        dm_g = -J_g * Sg * dt
        dm_b = -J_b * Sb * dt
        
        m_g += dm_g
        m_b += dm_b
        
        # --- C. Inverse Geometry Problem ---
        # We need to find [rg, rb] that match the new masses m_g, m_b
        target_Vg = m_g / rho0
        target_Vb = m_b / rho0
        
        def residuals(x):
            # x is [rg, rb] in microns
            # Penalize unphysical geometries to guide solver
            if x[1] >= x[0] or x[1] <= 1e-5: return [1e5, 1e5]
            
            v_g_new, v_b_new, _, _, _ = get_geometry_scaled(x)
            
            # Normalized Error (Fractional)
            # (Calculated - Target) / Target
            err_g = (v_g_new - target_Vg) / target_Vg
            err_b = (v_b_new - target_Vb) / target_Vb
            return [err_g, err_b]
        
        # 'hybr' is robust for systems of non-linear equations
        sol = root(residuals, [rg_um, rb_um], method='hybr', tol=1e-4)
        
        if sol.success:
            rg_um, rb_um = sol.x
        else:
            # If solver fails, simulation likely reached geometric limit
            break
            
        # Record Data
        current_time += dt
        t_vals.append(current_time)
        ratio_vals.append(rb_um / rg_um)
        
        # Stop condition (coalescence)
        if rb_um / rg_um > 0.65: break 
        
    return t_vals, ratio_vals

# ==========================================
# 4. PLOTTING (Replicating Figure 3)
# ==========================================
plt.figure(figsize=(10, 7))

# Simulate the three sizes mentioned in typical sintering papers
grain_sizes = [1e-6] 
colors = ['blue', 'orange', 'green']

for r0, col in zip(grain_sizes, colors):
    t_data, r_data = run_simulation(r0)
    
    # Plot
    label_str = f'$r_g = {r0*1e6:.0f} \mu m$'
    plt.loglog(t_data, r_data, label=label_str, color=col, linewidth=2)

plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('Neck Ratio $x = r_b / r_g$', fontsize=12)
plt.title(f'Sintering of Europa Ice (LunaIcy Simulation)\nT = {T_celsius} °C, $\\alpha$ = {alpha}', fontsize=14)
plt.grid(True, which="both", ls="-", alpha=0.4)
plt.legend(fontsize=12)

# Optional: Add slope guide for visual verification of power law
# Sintering usually follows x ~ t^(1/n). A slope of 1/5 or 1/6 is common for vapor transport.
# plt.loglog([10, 1000], [0.3, 0.3 * (1000/10)**(1/5)], 'k--', label='Slope ~ 1/5')

plt.tight_layout()
plt.show()