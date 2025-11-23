import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root
from scipy.integrate import quad

# --- 1. Constants ---
R = 8.314          
M = 0.018015       
rho0 = 917.0       
gamma = 0.065      
P0 = 3.5e12        
Q_sub = 51000.0    
alpha = 0.03       
T_celsius = -3.0
T = 273.15 + T_celsius
phi = 0.4          

# Scaling
L_S = 1e-6
V_S = L_S**3
A_S = L_S**2

# --- 2. Geometry (LunaIcy) ---

def get_rp(rg, rb):
    # rb, rg in MICRONS
    if rg <= rb: return 1e-9
    # Formula: rp = rb^2 / (2*(rg - rb))
    val = (rb**2) / (2 * (rg - rb))
    return max(1e-12, val) # Clamp to avoid div/0

def get_geometry_log(params):
    """
    Input: [ln(rg), ln(rb)]
    Output: Vg, Vb, Sg, Sb, rp, Kg, Kb (SI UNITS)
    """
    rg_um = np.exp(params[0])
    rb_um = np.exp(params[1])
    
    # Geometric Constraints for stability
    if rb_um >= rg_um * 0.99: rb_um = rg_um * 0.99
    
    rp_um = get_rp(rg_um, rb_um)
    
    # x* Intersection
    def h_diff(x):
        hg = np.sqrt(max(0, rg_um**2 - x**2))
        hb = (rb_um + rp_um) - np.sqrt(max(0, rp_um**2 - (rg_um - x)**2))
        return hg - hb
    
    # Approx x* is usually sufficient and more stable for derivatives
    # But we try precise root finding
    try:
        from scipy.optimize import brentq
        x_star_um = brentq(h_diff, 0, rg_um * 0.9999)
    except:
        x_star_um = rg_um * 0.9

    # Sg (Grain Area)
    def int_Sg(theta):
        val = 1 - (x_star_um / (rg_um * np.cos(theta)))**2
        return np.sqrt(max(0, val))
    
    limit = np.arccos(min(1.0, x_star_um/rg_um))
    val_Sg, _ = quad(int_Sg, 0, limit)
    Sg_um2 = 4 * (rg_um**2 * (np.pi - 2 * val_Sg))
    
    # Sb (Bond Area)
    L_um = rg_um - x_star_um
    term_asin = np.arcsin(min(1.0, L_um/rp_um))
    Sb_um2 = 2 * np.pi * rp_um * ((rb_um + rp_um)*term_asin - L_um)
    
    # Vg (Grain Volume)
    def A_func(r):
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
    
    # Vb (Bond Volume)
    t1 = np.pi * (rb_um + rp_um)**2 * L_um
    t2 = 2*(rb_um + rp_um)*np.pi*((L_um/2)*np.sqrt(max(0, rp_um**2 - L_um**2)) + (rp_um**2/2)*term_asin)
    t3 = np.pi * rp_um**2 * L_um
    t4 = (np.pi/3) * L_um**3
    Vb_um3 = t1 - t2 + t3 - t4
    
    # Curvatures (SI Units)
    Kg = 2.0 / (rg_um * L_S)
    Kb = -1.0 / (rp_um * L_S) # Concave

    return (Vg_um3 * V_S, Vb_um3 * V_S, 
            Sg_um2 * A_S, Sb_um2 * A_S, 
            rp_um * L_S, Kg, Kb)

# --- 3. Implicit Solver (Log-Space) ---

def solve_step(log_r_prev, m_prev, dt):
    """
    Solves for log(r_next) using Implicit Euler.
    """
    mg_old, mb_old = m_prev
    
    P_sat = P0 * np.exp(-Q_sub / (R * T))
    Kelvin_C = (gamma * M) / (R * T * rho0)
    Flux_C = alpha * np.sqrt(M / (2 * np.pi * R * T))
    
    def residual(log_r):
        # Decode geometry
        if log_r[1] >= log_r[0]: return [1e5, 1e5] # Constraint: rb < rg
        
        Vg, Vb, Sg, Sb, rp, Kg, Kb = get_geometry_log(log_r)
        
        # 1. Physics (Exponential Form)
        P_Kg = P_sat * np.exp(Kelvin_C * Kg)
        P_Kb = P_sat * np.exp(Kelvin_C * Kb)
        
        # 2. Flux Balance (User's m_gas derivation simplified)
        # P_gas = Weighted Average
        P_gas = (P_Kg * Sg + P_Kb * Sb) / (Sg + Sb)
        
        # 3. Fluxes
        J_g = Flux_C * (P_Kg - P_gas)
        J_b = Flux_C * (P_Kb - P_gas)
        
        # 4. Mass Balance (Implicit)
        # m_new - m_old - dt*(-J*S) = 0
        res_g = (Vg * rho0) - (mg_old - J_g * Sg * dt)
        res_b = (Vb * rho0) - (mb_old - J_b * Sb * dt)
        
        # Normalize residuals by mass to keep solver in range ~1.0
        return [res_g / mg_old, res_b / mb_old]

    # Use 'lm' (Levenberg-Marquardt) as it's robust for stiff/scaling issues
    sol = root(residual, log_r_prev, method='lm', tol=1e-6)
    return sol.x if sol.success else None

# --- 4. Main Loop ---

def run_simulation(r0_meters):
    rg_um = r0_meters / L_S
    rb_um = rg_um * 0.1 # Start at 5% to avoid Angstrom-scale singularity
    
    # Start with TINY time steps
    dt = 1e-4 
    t_max = 2e6 if r0_meters > 50e-6 else 2e4
        
    t_vals = []
    ratio_vals = []
    current_time = 0
    
    # Initial State
    log_r = [np.log(rg_um), np.log(rb_um)]
    Vg, Vb, _, _, _, _, _ = get_geometry_log(log_r)
    m_g = Vg * rho0
    m_b = Vb * rho0
    
    next_record = dt

    while current_time < t_max:
        
        new_log_r = solve_step(log_r, [m_g, m_b], dt)
        
        if new_log_r is None:
            # If implicit solve fails, try cutting dt
            dt *= 0.1
            print(f"Solver failed at t={current_time}, reducing dt to {dt}")
            if dt < 1e-9: break
            continue
            
        log_r = new_log_r
        
        # Update Masses
        Vg, Vb, _, _, _, _, _ = get_geometry_log(log_r)
        m_g = Vg * rho0
        m_b = Vb * rho0
        
        current_time += dt
        
        # Recording & Adaptive dt
        if current_time >= next_record:
            t_vals.append(current_time)
            ratio_vals.append(np.exp(log_r[1]) / np.exp(log_r[0]))
            
            next_record = current_time * 1.1 
            dt *= 1.1 # Safe to grow dt once we pass the initial snap
            
        if np.exp(log_r[1])/np.exp(log_r[0]) > 0.5: break
        
    return t_vals, ratio_vals

# --- 5. Plot ---
plt.figure(figsize=(10, 7))

sizes = [10e-6]
cols = ['#1f77b4', '#ff7f0e', '#2ca02c']

for r0, c in zip(sizes, cols):
    print(f"Simulating r0={r0}...")
    t, r = run_simulation(r0)
    plt.loglog(t, r, label=f'$r_g = {r0*1e6:.0f} \\mu m$', color=c, lw=2)

plt.xlabel('Time (s)')
plt.ylabel('Neck Ratio $x = r_b / r_g$')
plt.title(f'Sintering of Europa Ice (Implicit Log-Solver)\nT = {T_celsius} °C, $\phi={phi}$')
plt.grid(True, which="both", alpha=0.3)
plt.legend()
plt.show()