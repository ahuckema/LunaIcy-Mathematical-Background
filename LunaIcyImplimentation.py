import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import sys

# --- 1. Constants Block ---
# --- Physical Constants ---
R_GAS = 8.314                # J / (mol K) - Ideal gas constant
SIGMA_SB = 5.67e-8            # W / (m^2 K^4) - Stefan-Boltzmann constant
RHO_ICE = 917.0               # kg / m^3 - Density of solid ice
CP_ICE = 2108.0               # J / (kg K) - Specific heat of ice (constant approx.)
K_ICE = 2.3                   # W / (m K) - Thermal conductivity of solid ice

# --- Sintering Physics (from "LunaIcy" & "MultIHeaTS") ---
P_SAT_A = 28.87               # unitless
P_SAT_B = 6141.7              # K (related to Delta H_sub / R)

# *** FIX 1: RE-TUNED CONSTANTS ***
# My previous guesses (1e-24, 1e-22) were all far too small
# because the p_sat(T) term at 139K is ~1e-20.
# We need a much, much larger constant to compensate.
C_GP = 1e-11                  # (kg K^1.5) / (m^2 s Pa) - Tuned for this problem
C_PB = 1e-11                  # (kg K^1.5 m) / (m^2 s Pa) - Tuned for this problem
ALPHA_K = 0.5                 # Exponent for k(y) model
PHI_INITIAL = 0.4             # Initial porosity (loose sphere packing)

# --- Simulation & Planetary Parameters ---
DEPTH_D = 100.0                # m - Total depth of the simulation domain
N_GRID = 500                  # Number of grid points
H_GRID = DEPTH_D / (N_GRID - 1) # m - Grid spacing

# Boundary Condition Parameters
EPSILON = 0.95                # Emissivity of the ice surface
ALBEDO = 0.6                  # Bond albedo of Europa
F_SUN_JUPITER = 50.4          # W / m^2 - Mean solar flux at Jupiter
F_SOLAR_AVG = (1.0 - ALBEDO) * F_SUN_JUPITER  # ~20.16 W/m^2

# Simulation Time
T_YEAR = 3600.0 * 24.0 * 365.25 # Seconds in one year
T_SPAN = (0.0, 1e6 * T_YEAR)    # 1 Million Years

# We only need the start and end points for this plot
T_EVAL = np.array([T_SPAN[0], T_SPAN[1]])

# Epsilon for numerical stability
EPS_NUM = 1e-15


# --- 2. Physics & Coupling Functions ---
# (These are all correct)

def p_sat_func(T_clipped):
    """Calculates saturation vapor pressure (Pa)"""
    # Assumes T is already clipped
    return np.exp(P_SAT_A - P_SAT_B / T_clipped)

def porosity_func(rg, rb):
    """Porosity phi(y) model"""
    ratio = np.clip(rb / (rg + EPS_NUM), 0, 1)
    return PHI_INITIAL * (1.0 - ratio)

def rho_func(rg, rb):
    """Bulk density rho(y) (kg/m^3)"""
    phi = porosity_func(rg, rb)
    return (1.0 - phi) * RHO_ICE

def cp_func(rg, rb):
    """Specific heat cp(y) (J/kg/K)"""
    return CP_ICE # Constant

def k_func(rg, rb):
    """Thermal conductivity k(y) (W/m K)"""
    phi = porosity_func(rg, rb)
    ratio = np.clip(rb / (rg + EPS_NUM), 0, 1)
    k_min = 0.02 # Approx conductivity of vacuum
    k_eff = k_min + (K_ICE - k_min) * (1 - phi) * (ratio**ALPHA_K)
    return k_eff

# --- 3. Sintering Rate Function (f(y, T)) ---

def sintering_rate(y, T):
    """
    Calculates the sintering rate [drg/dt, drb/dt]
    y = [rg, rb]
    """
    
    # The NaN-Killer: Clamp T at a physical minimum
    T_clipped = np.maximum(T, 50.0)
    
    rg, rb = y
    
    # The physical brake
    ratio = rb / (rg + EPS_NUM)
    if ratio >= 1.0 or rg <= EPS_NUM:
        return [0.0, 0.0]
    
    # Add a smooth limiting factor
    limiter = (1.0 - ratio)**2 
    
    # --- Build Jacobian J ---
    J_11 = 4.0 * np.pi * (rg**2)
    J_12 = 0.0
    J_21 = -np.pi * (rb**4) / (2.0 * (rg**2 + EPS_NUM))
    J_22 = 2.0 * np.pi * (rb**3) / (rg + EPS_NUM)
    
    det_J = J_11 * J_22 - J_12 * J_21
    
    if np.abs(det_J) < EPS_NUM:
        return [0.0, 0.0]

    # --- Build Source Vector S ---
    # Use T_clipped for ALL T-dependent terms
    psat = p_sat_func(T_clipped)
    T_term = T_clipped**(-1.5)
    psat_T_term = psat * T_term
    
    J_gp = C_GP * psat_T_term * rg
    J_pb = C_PB * psat_T_term * (rg**2) / (rb + EPS_NUM)
    
    S_1 = (-J_gp / RHO_ICE)
    S_2 = (J_pb / RHO_ICE)
    
    # --- Solve f = J_inv * S ---
    drg_dt = (1.0 / det_J) * (J_22 * S_1 - J_12 * S_2)
    drb_dt = (1.0 / det_J) * (-J_21 * S_1 + J_11 * S_2)
    
    # Apply the limiter to the final rates
    drg_dt *= limiter
    drb_dt *= limiter
            
    return [drg_dt, drb_dt]


# --- 4. The Main ODE System (for solve_ivp) ---
# (This section is identical and correct)

def ode_system(t, U_flat):
    """
    This is the function for the ODE solver, implementing YOUR
    ghost-cell method and quasi-static approximation.
    U_flat is a 1D vector: [rg0, rb0, T0, rg1, rb1, T1, ...]
    """
    
    # Reshape the 1D state vector into a 2D N_GRID x 3 array
    # U = [ [rg_0, rb_0, T_0], [rg_1, rb_1, T_1], ... ]
    U = U_flat.reshape((N_GRID, 3))
    rg = U[:, 0]
    rb = U[:, 1]
    T = U[:, 2]
    
    # Create the 1D derivative vector, initialized to zeros
    dUdt_flat = np.zeros_like(U_flat)
    dUdt = dUdt_flat.reshape((N_GRID, 3))
    drg_dt = dUdt[:, 0]
    drb_dt = dUdt[:, 1]
    dT_dt = dUdt[:, 2]

    # --- A. Calculate Physics Properties at all nodes ---
    k_nodes = k_func(rg, rb)
    rho_nodes = rho_func(rg, rb)
    cp_nodes = cp_func(rg, rb)
    inv_rho_cp = 1.0 / (rho_nodes * cp_nodes + EPS_NUM)
    
    # --- B. Sintering ODEs (Loop over all nodes) ---
    s_rates = [sintering_rate(U[i, 0:2], U[i, 2]) for i in range(N_GRID)]
    s_rates = np.array(s_rates)
    drg_dt[:] = s_rates[:, 0]
    drb_dt[:] = s_rates[:, 1]

    # --- C. PDE (Loop over ALL nodes, 0 to N-1) ---
    # We use your GHOST-CELL method.
    for i in range(N_GRID):
        
        # --- Define k at half-points k_{i+1/2} and k_{i-1/2} ---
        if i < N_GRID - 1:
            k_plus = 0.5 * (k_nodes[i] + k_nodes[i+1])
        else:
            k_plus = k_nodes[i] # At i=d, k_{d+1/2} approx as k_d
            
        if i > 0:
            k_minus = 0.5 * (k_nodes[i] + k_nodes[i-1])
        else:
            k_minus = k_nodes[i] # At i=0, k_{-1/2} approx as k_0

        # --- Define T at neighbor points T_{i+1} and T_{i-1} ---
        if i == 0:
            # Surface: Calculate T_{-1} from your formula
            T_plus = T[i+1]
            T_m1_ghost = T[i+1] + (2 * H_GRID / (k_nodes[i] + EPS_NUM)) * \
                         (F_SOLAR_AVG - EPSILON * SIGMA_SB * T[i]**4)
            T_minus = T_m1_ghost
        
        elif i == N_GRID - 1:
            # Bottom: Calculate T_{d+1} from your formula
            T_p1_ghost = T[i-1] # T_{d+1} = T_{d-1}
            T_plus = T_p1_ghost
            T_minus = T[i-1]
            
        else:
            # Interior: Standard neighbors
            T_plus = T[i+1]
            T_minus = T[i-1]

        # --- Calculate Fluxes and dT/dt ---
        flux_plus = k_plus * (T_plus - T[i])
        flux_minus = k_minus * (T[i] - T_minus)
        
        dT_dt[i] = inv_rho_cp[i] * (flux_plus - flux_minus) / (H_GRID**2)

    return dUdt_flat

# --- 5. Setup and Run Simulation ---
# (This section is identical and correct)

print("Setting up initial conditions...")
sys.stdout.flush()

x_grid = np.linspace(0, DEPTH_D, N_GRID)

U_initial_flat = np.zeros(N_GRID * 3)
U_initial = U_initial_flat.reshape((N_GRID, 3))

U_initial[:, 0] = 1e-4  # rg: 100 micron grains
U_initial[:, 1] = 1e-6  # rb: 1 micron bonds (very small)
U_initial[:, 2] = 100.0 # T: 100K everywhere

print(f"Initial state at surface: rg={U_initial[0,0]}, rb={U_initial[0,1]}, T={U_initial[0,2]}")
print(f"Grid spacing h = {H_GRID:.4f} m")

print("Solving ODE system (this will take a moment)...")
sys.stdout.flush()
sol = solve_ivp(
    ode_system, 
    T_SPAN, 
    U_initial_flat, 
    method='BDF', 
    t_eval=T_EVAL
)
print("Simulation complete.")
sys.stdout.flush()

# --- 6. Process and Plot Results (Depth Profile à la Fig. 7) ---
# (This plotting section is identical to the last one, with one fix)

if not sol.success:
    print(f"Solver failed: {sol.message}")
else:
    print("Processing depth-profile data...")
    
    # Get the initial state
    U_initial_profile = U_initial_flat.reshape((N_GRID, 3))
    # Get the final state (last time point)
    U_final_profile = sol.y[:, -1].reshape((N_GRID, 3))

    # Extract initial and final bond radius (in meters)
    rb_initial_m = U_initial_profile[:, 1]
    rb_final_m = U_final_profile[:, 1]
    
    # Convert to micrometers (μm) for plotting
    rb_initial_microns = rb_initial_m * 1e6
    rb_final_microns = rb_final_m * 1e6

    print(f"Initial rb at surface: {rb_initial_microns[0]:.2f} μm")
    print(f"Final rb at surface:   {rb_final_microns[0]:.2f} μm")
    print(f"Final rb at 3m depth:  {rb_final_microns[int(3/H_GRID)]:.2f} μm")
    
    # --- Create the Plot (like the image you sent) ---
    fig, ax = plt.subplots(figsize=(7, 8))
    fig.suptitle('Bond Radius Evolution (Fig. 7 style)', fontsize=16)

    # Plot initial bond radius (t=0)
    ax.plot(rb_initial_microns, x_grid, 'b--', label='$r_b(x, 0)$')
    
    # Plot final bond radius (t=1 Myr)
    ax.plot(rb_final_microns, x_grid, 'b-', linewidth=2, label='$r_b(x, t_f)$')
    
    # --- Set Axes ---
    # *** FIX 2: Added r'' to fix SyntaxWarning ***
    ax.set_xlabel(r'Bond Radius ($\mu\text{m}$)', fontsize=14)
    ax.set_ylabel('Depth (m)', fontsize=14)
    
    # Set Y-axis to log scale
    ax.set_yscale('log')
    
    # Invert Y-axis so 0 is at the top
    ax.invert_yaxis()
    
    # Set y-limits to be similar to the paper's plot
    ax.set_ylim(DEPTH_D, 1e-3)
    # Set x-limits to be reasonable (e.g., 0 to 100 microns)
    ax.set_xlim(left=0, right=100) # 100 microns is the grain size
    
    # Add gridlines
    ax.grid(True, which="both", ls="--", alpha=0.7)
    
    # Add legend
    ax.legend(fontsize=14)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()