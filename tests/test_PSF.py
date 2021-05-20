import os
import sys
from pathlib import Path
import logging
import time
import numpy as np
from casadi import vertcat
from tqdm import tqdm

from PSF.utils import formulate_center_problem, Hh_from_disconnected_constraints, polytope_center, solve

HERE = Path(__file__).parent
sys.path.append(HERE.parent)  # to import gym and psf
os.chdir(HERE.parent)
from PSF.PSF import PSF
import gym_rl_mpc.objects.symbolic_model as sym
import gym_rl_mpc.utils.model_params as params

logging.basicConfig(level=logging.DEBUG)

print("Test Started")
number_of_state_perm = 20
number_of_input_perm = 20
np.random.seed(42)
init_psf_args = dict(sys=sym.get_sys(),
                     N=200,
                     T=100,
                     t_sys=sym.get_terminal_sys(),
                     R=np.diag([
                         1 / params.max_thrust_force ** 2,
                         1 / params.max_blade_pitch ** 2,
                         1 / params.max_power_generation ** 2
                     ]),
                     PK_path=Path(HERE, "terminalset"),
                     ext_step_size=0.1,
                     # slew_rate=[1e6, 8 * params.DEG2RAD, 1e6],
                     terminal_type="steady"
                     )

psf = PSF(**init_psf_args)
args_list = []
nx = sym.x.shape[0]
nu = sym.u.shape[0]
start = time.time()
for j in range(number_of_state_perm):

    x = [np.random.uniform(low=-7, high=7) * params.DEG2RAD,
         0,
         np.random.uniform(low=5.1, high=7.5) * params.RPM2RAD]
    w = np.random.uniform(low=11 * 2 / 3, high=25 * 2 / 3)
    for i in range(number_of_input_perm):
        u_L = [
            np.random.uniform(low=-params.max_thrust_force, high=params.max_thrust_force),
            np.random.uniform(low=-4 * params.DEG2RAD, high=params.max_blade_pitch),
            np.random.uniform(low=0, high=params.max_power_generation)
        ]
        kwargs_calc = dict(x=x,
                           u_L=u_L,
                           u_prev=None,
                           ext_params=w,
                           reset_x0=False,
                           )

        args_list.append(kwargs_calc)

Hz, hz = Hh_from_disconnected_constraints(np.vstack([sym.sys_lub_x, sym.sys_lub_u]))

z0 = polytope_center(Hz, hz)
solver, lbg, ubg = formulate_center_problem(sym.symbolic_x_dot, vertcat(sym.x, sym.u), Hz, hz, p=sym.w)
t_sys = sym.get_terminal_sys()

for kwargs in tqdm(args_list[:]):
    # print(sym.solve_initial_problem(kwargs["ext_params"]))
    z0 = solve(solver, lbg, ubg, 1, kwargs["ext_params"])
    x0, u0 = np.vsplit(z0, [sym.x.shape[0]])
    kwargs["u_prev"] = u0
    t_sys["hv"][-2] = -(round(kwargs["ext_params"]) - 1)
    t_sys["hv"][-1] = round(kwargs["ext_params"]) + 1
    psf.calculate_new_terminal(new_t_sys=t_sys)
    u = psf.calc(**kwargs)
    # logging.info(u)

end = time.time()
number_of_iter = number_of_input_perm * number_of_state_perm
print(f"Number of state permutations: {number_of_state_perm}\n Number of input permutations: {number_of_input_perm}")
print(f"Solved {number_of_iter} iterations in {end - start} s. {(end - start) / number_of_iter} s/step]")
