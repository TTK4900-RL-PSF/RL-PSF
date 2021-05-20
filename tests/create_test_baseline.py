import pickle
import git
from pathlib import Path
import numpy as np
from casadi import vertcat
from tqdm import tqdm

import gym_rl_mpc.objects.symbolic_model as sym
import gym_rl_mpc.utils.model_params as params
from PSF.PSF import PSF
from PSF.utils import solve, Hh_from_disconnected_constraints, polytope_center, formulate_center_problem

HERE = Path(__file__).parent
init_psf = dict(sys="sym.get_sys()",
                N=20,
                T=10,
                t_sys="sym.get_terminal_sys()",
                R=np.diag([
                    1 / params.max_thrust_force ** 2,
                    1 / params.max_blade_pitch ** 2,
                    1 / params.max_power_generation ** 2
                ]),
                PK_path=Path(HERE, "terminalset"),
                ext_step_size=0.1,
                slew_rate=[1e6, 8 * params.DEG2RAD, 1e6],
                )

number_of_state_perm = 20
number_of_input_perm = 20
args_list = []
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
                           reset_x0=False
                           )

        args_list.append(kwargs_calc)

repo = git.Repo(search_parent_directories=True)
sha = repo.head.object.hexsha

test_case = dict(
    git_hex=sha,
    psf_init_kwargs=init_psf,
    list_calc_kwargs=args_list
)

pickle.dump(test_case, open("test_case.dat", "wb"))

print("Assumes sym.get_sys and sym.get_terminal_sys is static across time and tests")
test_case=pickle.load(open("test_case.dat", "rb"))

psf_init_kwargs = test_case["psf_init_kwargs"]
list_calc_kwargs = test_case["list_calc_kwargs"]

Hz, hz = Hh_from_disconnected_constraints(np.vstack([sym.sys_lub_x, sym.sys_lub_u]))

z0 = polytope_center(Hz, hz)
solver, lbg, ubg = formulate_center_problem(sym.symbolic_x_dot, vertcat(sym.x, sym.u), Hz, hz, p=sym.w)

psf = PSF(**psf_init_kwargs)
for kwargs in tqdm(args_list[:]):
    # print(sym.solve_initial_problem(kwargs["ext_params"]))
    z0 = solve(solver, lbg, ubg, 1, kwargs["ext_params"])
    x0, u0 = np.vsplit(z0, [sym.x.shape[0]])
    kwargs["u_prev"] = u0
    u = psf.calc(**kwargs)
    # logging.info(u)

