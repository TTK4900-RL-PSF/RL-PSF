import numpy as np

RAD2DEG = 180/np.pi
DEG2RAD = np.pi/180
RAD2RPM = 60/(2*np.pi)
RPM2RAD = (2*np.pi/60)

def power_regime(wind_speed):
    if wind_speed < 3:
        return 0
    elif wind_speed < 10.59:
        return (((wind_speed-3)/(10.59-3))**2)
    else:
        return 1


def omega_setpoint(wind_speed):
    if wind_speed < 6.98:
        return 5*RPM2RAD       # 5 rpm
    elif wind_speed < 10.59:
        return (((7.55-5)/(10.59-6.98))*(wind_speed-6.98)+5)*RPM2RAD
    else:
        return 7.55*RPM2RAD    # 7.55 rpm

# Platform parameters
L = 144.45
L_thr = 50

# Rotor parameters
B = 0.97                                # Tip loss parameter From 15MW turbine documentation
lambda_star = 9                         # From 15MW turbine documentation
C_P_star = 0.489                        # From 15MW turbine documentation
C_F = 0.8                               # From 15MW turbine documentation
rho = 1.225                             # Air density
R = 120                                 # Blade length From 15MW turbine documentation
A = np.pi*R**2
R_p = B*R
A_p = np.pi*R_p**2
k_r = (2*rho*A_p*R)/(3*lambda_star)     # k in Force/Torque equations
l_r = (2/3)*R_p                         # l in Force/Torque equations
b_d_r = 0.5*rho*A*(B**2*(16/27)-C_P_star)*(R/lambda_star)**3  # b_d in Force/Torque equations
d_r = 0.5*rho*A*C_F                     # d in Force/Torque equations
J_r = 4.068903574982517e+07             # From OpenFast
wind_inflow_ratio = 2/3

max_thrust_force = 5e5                  # maximum force input [N]
max_thrust_rate = 1.5e5
max_blade_pitch = 20*(np.pi/180)        # maximum deviation from beta^*
min_blade_pitch_ratio = 0.2 
max_wind_force = 5e7                    # Just used for animation scaling
max_blade_pitch_rate = 8*(np.pi/180)    # Max blade pitch rate [rad/sec]
max_power_generation = 15e6
max_power_rate = 5e6
max_generator_torque = max_power_generation/omega_setpoint(0)
tau_thr = 2                             # Thrust force time constant
tau_blade_pitch = 1.3                   # Blade pitch time constant
tau_power = 2                           # Power generator time constant


C_1 = 4.4500746068705328
C_2 = -4.488263864070078
C_3 = 0.0055491593253495
C_4 = -6.86458290065766e-12*L_thr
C_5 = 0.000000000991589



