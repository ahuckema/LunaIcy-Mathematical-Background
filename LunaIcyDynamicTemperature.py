import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
from scipy.optimize import root

R_gas = 8.314         
M_h2o = 0.018015     
rho0 = 917.0          
gamma = 0.065         
P0 = 3.5e12           
Q_sub = 51000.0       
sigma_SB = 5.67e-8    
cp_ice = 2100.0       
epsilon =0.9         

alpha = 0.03          
phi = 0.4             
Albedo = 0.0          
G_sc = 50.0           
dist_sun = 1.0        

dt = 3600.0           
day_len = 3.55 * 86400 
t_max = 100 * 86400   # 100 Days
depth = 0.2           
Nx = 40               
dx = depth / Nx
x_grid = np.linspace(dx/2, depth - dx/2, Nx)

LS = 1e-6 

def get_geometry_state(r_g, r_b):
    """
    Volumes, surface area and curvature are calculated here
    """
    rg_u = r_g / LS
    rb_u = r_b / LS
    
    # Stability clamp
    if rb_u >= 0.99 * rg_u: 
        rb_u = 0.99 * rg_u
    
    rp_u = (rb_u**2) / (2 * (rg_u - rb_u))
    
    #Find \hat{x}^* by solving the Eq. numerically
    from scipy.optimize import brentq
    def h_diff(x):
        return np.sqrt(max(0, rg_u**2 - x**2)) - (rb_u + rp_u - np.sqrt(max(0, rp_u**2 - (rg_u - x)**2)))
    try:
        x_star_u =brentq(h_diff, 0, rg_u*0.999)
    except:
        x_star_u = rg_u*0.99

    # Volumes 
    term_cap =(np.pi * (rg_u -x_star_u)**2 / 3.0) * (2* rg_u + x_star_u)
    Vg_u = (4.0/3.0) * np.pi * rg_u**3 - term_cap
    
    
    #Bond Volumes
    from scipy.integrate import quad
    def bond_area(x1):
        if x1 > rp_u: return np.pi * (rb_u + rp_u)**2
        r_loc =   rb_u + rp_u - np.sqrt(max(0, rp_u**2 - x1**2))
        return np.pi * r_loc**2
    
    limit = rg_u - x_star_u
    V_b_half, _ = quad(bond_area, 0, limit, epsabs=1e-12, epsrel=1e-12)
    Vb_u = 2 * V_b_half 

    #Surfaces
    Sg_u= 2 * np.pi * rg_u * (rg_u + x_star_u)
    
    h_val = rg_u - x_star_u
    asin_term = np.arcsin(min(1.0, h_val / rp_u))
    Sb_u = 2 * np.pi * rp_u * ((rb_u + rp_u)*asin_term - h_val)

    # Curvatures
    Kg = 2.0 / r_g
    Kb = -1.0 / r_b 

    return Vg_u*(LS**3), Vb_u*(LS**3), Sg_u*(LS**2), Sb_u*(LS**2), Kg, Kb

def get_fluxes(r_g, r_b, T):
    Vg, Vb, Sg, Sb, Kg, Kb = get_geometry_state(r_g, r_b)
    
    P_sat = P0 * np.exp(-Q_sub / (R_gas * T))
    kelvin_coef = (gamma * M_h2o) / (R_gas *T * rho0)
    
    P_Kg = P_sat *(1 +kelvin_coef * Kg)
    P_Kb = P_sat* (1 + kelvin_coef * Kb)
    
    term_bracket = (Sg+ Sb) + kelvin_coef * (Sg * Kg   + Sb * Kb)
    numerator = P_sat *  term_bracket
    denominator = (1 -phi) * R_gas * T * (Sg + Sb)
    m_gas = numerator * (M_h2o * Vg *  phi) /denominator
    P_gas = ((1 - phi) * m_gas * R_gas *T)/ (M_h2o *  Vg * phi)
    
    kinetics = alpha * np.sqrt(M_h2o / (2 * np.pi *R_gas * T))
    J_g = kinetics * (P_Kg-P_gas)
    J_b = kinetics * (P_Kb - P_gas)
    
    return J_g, J_b, Sg,Sb



def solve_microstructure_increment(rg, rb, T, dt_step):
    """
    Solve Implicit Euler Scheme
    """
    
    # Initial state
    Vg_old, Vb_old, _, _, _, _ =get_geometry_state(rg, rb)
    m_g_old = Vg_old * rho0
    m_b_old = Vb_old * rho0

    # Calculate errors
    def residual(x):
        r_g_curr, r_b_curr=x
        #Physically valid
        if r_g_curr <= 1e-7 or r_b_curr <= 1e-7:
            return [100.0, 100.0] 
            
        Vg_new,Vb_new, Sg_new, Sb_new, _, _ =   get_geometry_state(r_g_curr, r_b_curr)
        Jg_new, Jb_new, _, _= get_fluxes(r_g_curr, r_b_curr,T)
        
        m_g_new = Vg_new * rho0
        m_b_new = Vb_new * rho0
        
        res_g = (m_g_new -m_g_old + Jg_new * Sg_new  * dt_step) /m_g_old
        res_b = (m_b_new - m_b_old  + Jb_new* Sb_new * dt_step) / m_b_old
        
        return [res_g, res_b]

    #Solve non-Linear eq.
    sol = root(residual,[rg, rb], method='hybr', tol=1e-6)
    
    # Fallback Handling
    if not sol.success:
        return rg, rb
        
    return sol.x[0], sol.x[1]


def calc_conductivity(T, rg, rb):
    k0 = 567.0 / T 
    return k0 * (1 - phi) * (rb / rg)

def solve_heat_implicit(T_old,rg_arr, rb_arr, dt_step, time_curr):
    T_new = T_old.copy()
    N =len(T_old)
    coeff = dt_step / (rho0 * (1-phi) * cp_ice)
    inv_h2 = 1.0 /   (dx**2)
    
    #Here PDE is solved
    for _ in range(3): 
        k_vals = calc_conductivity(T_new, rg_arr, rb_arr)
        G = np.zeros(N)
        diag, upper, lower =np.zeros(N ), np.zeros(N), np.zeros(N)
        
        theta_i = (2 * np.pi * time_curr) / day_len
        F_sol = max(0.0, (1- Albedo) * (G_sc / dist_sun**2) * np.cos(theta_i))
        Rad_term = epsilon * sigma_SB * T_new[0]**4
        T_ghost = (2*dx / k_vals[0])*(F_sol - Rad_term) + T_new[1]
        dTg_dT0 = (2*dx /  k_vals[0]) * (-4 * epsilon * sigma_SB * T_new[0]**3)
        
        kp, km = 0.5 * (k_vals[1] + k_vals[0]), k_vals[0]
        G[0] = T_new[0] - T_old[0] - coeff * (kp*(T_new[1]-T_new[0]) - km*(T_new[0]-T_ghost))*inv_h2
        diag[0] = 1.0 - coeff * (-kp- km*(1.0-dTg_dT0))*inv_h2
        upper[1] = - coeff * (kp + km)*inv_h2
        
        for i in range(1, N-1):
            kp, km = 0.5*(k_vals[i+1]+k_vals[i]), 0.5*(k_vals[i]+k_vals[i-1])
            G[i] = T_new[i] - T_old[i] - coeff * (kp*(T_new[i+1]-T_new[i]) - km*(T_new[i]-T_new[i-1]))*inv_h2
            diag[i] = 1.0 - coeff * (-kp - km)*inv_h2
            upper[i+1] = - coeff * kp*inv_h2
            lower[i] = - coeff * km*inv_h2

        i =   N-1
        km = 0.5*(k_vals[i]+k_vals[i-1])
        G[i] = T_new[i]-T_old[i] - coeff * (-km*(T_new[i]-T_new[i-1]))*inv_h2
        diag[i] = 1.0 - coeff * (-km)*inv_h2
        lower[i] = - coeff *km*inv_h2

        J_b = np.zeros((3, N))
        J_b[0, 1:]= upper[1:] 
        J_b[1, :] =    diag
        J_b[2, :-1] = lower[1:]
        T_new += scipy.linalg.solve_banded((1, 1), J_b, -G)
            




    return T_new

### Main Program here####
T_field = np.ones(Nx) * 110.0
rg_field = np.ones(Nx) * 100.0 * LS 
rb_field = np.ones(Nx) * 10.0 * LS 

rb_initial = rb_field.copy()
current_time = 0.0

while current_time<t_max:
    for i in range(Nx):
        rg_field[i], rb_field[i] = solve_microstructure_increment(
            rg_field[i], rb_field[i], T_field[i], dt
        )
        
    T_field = solve_heat_implicit(T_field, rg_field, rb_field, dt, current_time)
    current_time += dt



k_surf = calc_conductivity(T_field[0], rg_field[0], rb_field[0])
rho_eff = rho0 * (1 - phi)
omega = 2 * np.pi / day_len
skin_depth = np.sqrt( (2 * k_surf) / (rho_eff * cp_ice * omega) )





plt.figure(figsize=(8, 6))

plt.plot(rb_initial/LS, x_grid, color='steelblue', linestyle='--', linewidth=2, label='$r_b(x, 0)$')
plt.plot(rb_field/LS, x_grid, color='navy', linestyle='-', linewidth=2, label='$r_b(x, t_f)$')

plt.axhline(y=skin_depth, color='indianred', linestyle='--', linewidth=2, label='Diurnal skin depth')

plt.yscale('log')
plt.gca().invert_yaxis()
plt.xlabel(r'Bond Radius ($\mu m$)', fontsize=14)
plt.ylabel('Depth (m)', fontsize=14)
plt.title(f'Lattice Microstructure (t={t_max/86400:.0f} days)', fontsize=14)
plt.grid(True, which="both", alpha=0.5)
plt.legend(fontsize=12)

# Verify smoothness by printing the top 5 values
print("Top 5 bond radii (microns):", rb_field[:5]/LS)

plt.tight_layout()
plt.show()