import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from PSF.PSF import PSF
from gym_rl_mpc import DEFAULT_CONFIG

import gym_rl_mpc.objects.symbolic_model as sym
import gym_rl_mpc.utils.model_params as params

PI = 3.14
df = pd.read_csv(Path("logs", "debug", "crash_data.csv"))

states = ["theta", "theta_dot", "omega"]
p=[ "wind_speed", "adjusted_wind_speed"]

not_input = p + states
rad_to_deg_list = ["theta", "theta_dot"]
rad_to_rpm_list = ["omega"]
df[rad_to_deg_list] = df[rad_to_deg_list] * 360 / (2 * PI)
df[rad_to_rpm_list] = df[rad_to_rpm_list] * 60 / (2 * PI)

df[not_input].plot()


df.drop(labels=not_input, axis=1).plot()
plt.show()

sys = sym.get_sys()
t_sys = sym.get_terminal_sys()
R = np.diag(
    [
        1 / params.max_thrust_force ** 2,
        1 / params.max_blade_pitch ** 2,
        1 / params.max_power_generation ** 2
    ])
actuation_max_rate = [params.max_thrust_rate, params.max_blade_pitch_rate, params.max_power_rate]

## PSF init ##
psf = PSF(sys=sys, N=20, T=20, t_sys=t_sys, R=R, PK_path=Path("PSF", "stored_PK"), slew_rate=actuation_max_rate,
          ext_step_size=DEFAULT_CONFIG["step_size"], slack_flag=True)
df = pd.read_csv(Path("logs", "debug", "crash_data.csv"))
for i in range(df.shape[0]):
    row = df.iloc[i, :]
    F_thr = row["agent_F_thr"] * params.max_thrust_force
    blade_pitch = row["agent_blade_pitch"] * params.max_blade_pitch
    power = row["agent_power"] * params.max_power_generation
    action_un_normalized = [F_thr, blade_pitch, power]
    new_adjusted_wind_speed = params.wind_inflow_ratio * row["wind_speed"] - params.L * np.cos(row["theta"]) * row["theta_dot"]
    psf_params = [new_adjusted_wind_speed]
    u_prev = action_un_normalized
    args = dict(x=row[states].values,
                u_L=action_un_normalized,
                ext_params=psf_params,
                u_prev=u_prev)

    psf_corrected_action_un_normalized = psf.calc(**args)
