import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import root, brentq

#Constants
R = 8.314          
M = 0.018015       
rho0 = 917.0       
gamma = 0.065       
P0 = 3.5e12        
Q_sub = 51000.0    
sigma_SB = 5.67e-8 

#Parameters
alpha = 0.03       
phi = 0.4           
T_celsius = -3.0
T_val= 273.15 + T_celsius

# Scaling into Microns
LS =  1e-6          
AS = LS**2         
VS = LS**3         

def get_geometry(r_g_um, r_b_um):
    """
    First r_p is determined, then we solve for \hat{x}^* via Brent method, 
    then Grains surface and volumes are determined. Analogously for the bonds
    """
    #r_p
    if r_b_um >= r_g_um*0.999: r_b_um = r_g_um*0.999 # Stability clamp
    # Singularity check
    if r_b_um < 1e-6: r_p_um = 1e-6
    else: r_p_um = (r_b_um**2) / (2 * (r_g_um - r_b_um))
    
    # 2. \hat{x}^*
    def h_diff(x):
        hg = np.sqrt(max(0, r_g_um**2 - x**2))
        hb = (r_b_um +r_p_um) - np.sqrt(max(0, r_p_um**2 -(r_g_um - x)**2))
        return hg -hb
    
    try:
        x_star_um =brentq(h_diff, 0, r_g_um*0.999)
    except:
        x_star_um =  r_g_um * 0.9 # Fallback if geometry is degenerate

    #We use the Manifold integrals using Quadrature
    # A(x*, r) for arbitrary r<r_g
    def integrand_A (theta, r):
        # sqrt(1 - (x*/(r*cos(theta)))^2)
        val = 1 - (x_star_um / (r * np.cos(theta)))**2
        return np.sqrt(max(0, val))

    # Calculate A(x*, rg) for Surface
    limit_S  = np.arccos(min(1.0, x_star_um / r_g_um))
    A_val_S, _ =quad(integrand_A, 0, limit_S, args=(r_g_um,))
    
    # Surface:
    S_g_um2 = 4 * (r_g_um**2 * (np.pi - 2 * A_val_S))

    #Now For Volume:
    def rad_integrand(r):
        lim = np.arccos(min(1.0,x_star_um / r))
        # Here we do nested Quadrature for A(x*, r) and the integral over itself
        a_val, _ = quad(  integrand_A, 0, lim, args=(r,))
        return (r**2) *a_val

    #Volume of Grain:
    int_radial, _ = quad(rad_integrand, x_star_um, r_g_um)
    term1 =(np.pi *x_star_um**3) /3.0
    term2 = np.pi * ((r_g_um**3 / 3.0) -   ( x_star_um**3 / 3.0))
    V_g_um3 = 4 * (term1 + term2 - 2 * int_radial)

    ### Now the Bonds ###
    L_um = r_g_um- x_star_um 
    
    #Bond Surface Area
    term_asin = np.arcsin(min(1.0, L_um / r_p_um))
    S_b_single = 2* np.pi * r_p_um * ((r_b_um + r_p_um) * term_asin - L_um)
    S_b_um2 = 2 * S_b_single # Two bonds
    
    #The A in the calculation for the bond Volume
    term_sqrt =np.sqrt(max(0, r_p_um**2 - L_um**2))
    A_term = np.pi * ((L_um / 2) * term_sqrt + (r_p_um**2 / 2) * term_asin)
    
    # All the summmands from the calculation
    t1 = np.pi*(r_b_um + r_p_um)**2 * L_um
    t2 = 2 * (r_b_um + r_p_um) * A_term
    t3 = np.pi * r_p_um**2 * L_um
    t4 = (np.pi / 3) * L_um**3
    
    V_b_single =t1 -t2 + t3 - t4
    V_b_um3 = 2  *V_b_single # Two bonds

    # Curvature
    Kg = 2.0 /(r_g_um * LS)
    Kb = -1.0 / (r_p_um * LS) 

    return (V_g_um3 * VS, V_b_um3 * VS, 
            S_g_um2 * AS, S_b_um2 * AS, 
            Kg, Kb)

def get_mass_fluxes(state,T):
    """
    First m_gas calculated and then Fluxes Jg, Jb 
    """
    rg, rb = state
    # Clamp to prevent math domain errors in implicit solver steps
    if rg < 1e-4: rg = 1e-4
    if rb < 1e-6: rb = 1e-6
        
    Vg, Vb, Sg, Sb, Kg, Kb = get_geometry(rg,   rb)
    
    # Saturation Pressure on a flat surface
    P_sat =P0 * np.exp(-Q_sub / (R * T))
    
    # Kelvin Pressures 
    kelvin_factor = (gamma * M) / (R * T * rho0)
    P_Kg = P_sat * (1 + kelvin_factor * Kg)
    P_Kb = P_sat * (1 + kelvin_factor * Kb)
    
    # Equilibrium Gas Mass
    S_tot = Sg +  Sb
    kelvin_sum= Sg * Kg + Sb * Kb
    numerator = P_sat * (S_tot + kelvin_factor * kelvin_sum)
    geometric_factor = (M * Vg * phi) / ((1 - phi) * R * T * S_tot)
    m_gas = numerator * geometric_factor
    
    # Gas Pressure
    P_gas = ((1 - phi) * m_gas * R * T) / (M * Vg * phi)
    
    # Fluxes
    kinetics = alpha * np.sqrt(M / (2 * np.pi * R * T))
    
    J_g = kinetics * (P_Kg - P_gas)
    J_b = kinetics * (P_Kb - P_gas)
    
    return J_g, J_b,Sg, Sb, Vg, Vb

def solve_implicit_step(r_g_old,  r_b_old, dt, T, rho0):
    V_g_old, V_b_old, _, _, _, _ = get_geometry(r_g_old, r_b_old)
    m_g_old = V_g_old * rho0
    m_b_old = V_b_old *  rho0

    #Errors
    def residual(x):
        r_g_curr, r_b_curr = x
        
        # Enforce physical positivity 
        if r_g_curr <= 1e-5 or r_b_curr <= 1e-6:
            return [10.0, 10.0] # Penalty
            
        # Evaluate Fluxes and Geometry at the future
        J_g_new, J_b_new,S_g_new, S_b_new, V_g_new, V_b_new = get_mass_fluxes([r_g_curr, r_b_curr], T)
        
        m_g_new = V_g_new *rho0
        m_b_new= V_b_new * rho0
        
        # Normalize residual by mass to prevent scale-based solver failure (tol=1e-8 vs 1e-15 mass)
        res_g = (m_g_new - m_g_old + J_g_new * S_g_new * dt) / m_g_old
        res_b = (m_b_new - m_b_old + J_b_new * S_b_new * dt) / m_b_old
        
        return [res_g, res_b]

    # nonlinear system
    sol = root(residual, [r_g_old, r_b_old], method='hybr', tol=1e-6)
    
    if not sol.success:
         # Fallback to explicit step if implicit fails
        return [r_g_old,r_b_old]
        
    return sol.x

 #########Main Program #########
def run_simulation(r0_um, t_max):
    # Initial State
    rg_um = r0_um
    rb_um = r0_um * 0.1 # Smaller Bond
    
    current_time = 0
    dt = 1e-1 #Step size
    
    # History
    times =[]
    ratios = []
    
    next_record = 0
    while current_time < t_max:
        
        #implicit Euler
        new_radii = solve_implicit_step(rg_um, rb_um, dt, T_val, rho0)
        rg_um, rb_um = new_radii
        
        current_time += dt
        
        # Adaptive time step
        ratio = rb_um / rg_um
        if current_time >= next_record:
            times.append(current_time)
            ratios.append(ratio)
            next_record = max(dt, current_time * 1.1)
            
            # Accelerate time step as system relaxes
            dt *= 1.1
        
        if ratio > 0.6: break # Sintering limit
        
    return np.array(times), np.array(ratios)

plt.figure(figsize=(10, 6))
grain_sizes = [40, 100, 220] # Microns
colors = ['blue', 'orange', 'green']

for r0, col in zip(grain_sizes, colors):
    t, x = run_simulation(r0, 5e6)
    plt.loglog(t, x, label=rf'$r_g = {r0} \mu m$', color=col, linewidth=2)

plt.xlabel("Time (s)")
plt.ylabel(r"Neck Ratio $x = r_b / r_g$")
plt.title(rf"Reimplemented LunaIcy Sintering" "\n" rf"$T={T_celsius}^\circ C, \phi={phi}, \alpha={alpha}$")
plt.grid(True, which="both", alpha=0.3)
plt.legend()
plt.show()