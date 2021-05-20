import os
import sys
from pathlib import Path

import time
import numpy as np

HERE = Path(__file__).parent
sys.path.append(HERE.parent)  # to import gym and psf
os.chdir(HERE.parent)
from PSF.PSF import PSF
import gym_rl_mpc.objects.symbolic_model as sym
import gym_rl_mpc.utils.model_params as params

print("Test Started")
number_of_state_perm = 3
number_of_input_perm = 20

nx = sym.x.shape[0]
nu = sym.u.shape[0]

np.random.seed(42)
kwargs = dict(sys={"xdot": sym.symbolic_x_dot,
                   "x": sym.x,
                   "u": sym.u,
                   "p": sym.w,
                   "Hx": sym.Hx,
                   "hx": sym.hx,
                   "Hu": sym.Hu,
                   "hu": sym.hu
                   },
              N=200,
              T=20,
              R=np.diag([
                  1 / params.max_thrust_force ** 2,
                  1 / params.max_blade_pitch ** 2,
                  1 / params.max_power_generation ** 2
              ]),
              Q=np.diag([
                  0,
                  1,
                  0,
              ]),
              PK_path=Path(HERE, "terminalset"),
              ext_step_size=0.1,
              lin_bounds={"w": [3, 25],
                          "u_p": [0 * params.max_blade_pitch, params.max_blade_pitch],
                          "Omega": [params.omega_setpoint(3),
                                    params.omega_setpoint(25)],
                          "P_ref": [0, params.max_power_generation],
                          "theta": [-10 * sym.DEG2RAD, 10 * sym.DEG2RAD],
                          "theta_dot": [-45 * sym.DEG2RAD, 45 * sym.DEG2RAD]
                          },
              slew_rate=[1e5, 8 * params.DEG2RAD, 1e6],
              LP_flag=False,
              slack_flag=True,
              jit_flag=False,
              mpc_flag=True
              )

psf_HF = PSF(**kwargs)

kwargs["N"] = 20
psf_LF = PSF(**kwargs)

start = time.time()
for j in range(number_of_state_perm):
    x = [np.random.uniform(low=-7, high=9) * params.DEG2RAD,
         0,
         np.random.uniform(low=5.1, high=7.5) * params.RPM2RAD]
    ext_params = np.random.uniform(low=3, high=25)
    for i in range(number_of_input_perm):
        kwargs_calc = dict(x=x,
                           x_ref=[
                               2 * np.random.uniform(low=-params.max_thrust_force, high=params.max_thrust_force),
                               2 * np.random.uniform(low=-4 * params.DEG2RAD, high=params.max_blade_pitch),
                               2 * np.random.uniform(low=0, high=params.max_power_generation)
                           ],
                           u_prev=[0, 0, 0],
                           u_L=[0, 0, 0],
                           ext_params=ext_params,
                           reset_x0= True
                           )
        print(psf_LF.calc(**kwargs_calc))

end = time.time()
number_of_iter = number_of_input_perm * number_of_state_perm
print(f"Number of state permutations: {number_of_state_perm}\n Number of input permutations: {number_of_input_perm}")
print(f"Solved {number_of_iter} iterations in {end - start} s. {(end - start) / number_of_iter} s/step]")
