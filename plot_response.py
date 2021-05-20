import argparse
from plot_scripts.utils.agent_paths import agent_paths, agent_paths_psf
import gym_rl_mpc
import gym
from stable_baselines3 import PPO
import numpy as np
from utils import simulate_episode
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import os
from gym_rl_mpc.utils import model_params as params
from gym_rl_mpc.utils.model_params import RAD2DEG, RAD2RPM

def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--env',
        type=str,
        default="VariableWindPSFtestManual-v17",
        choices=gym_rl_mpc.SCENARIOS.keys(),
        help="Environment to run."
    )
    parser.add_argument(
        '--time',
        type=int,
        default=300,
        help='Max simulation time (seconds).',
    )
    parser.add_argument(
        '--save',
        help='Save plots',
        action='store_true'
    )
    parser.add_argument(
        '--psf_agent',
        help='Plot results of agents trained with PSF',
        action='store_true'
    )
    parser.add_argument(
        '--psf',
        help='Plot results with PSF on during testing',
        action='store_true'
    )
    parser.add_argument(
        '--save_sim_data',
        help='Plot results with PSF on during testing',
        action='store_true'
    )
    parser.add_argument(
        '--wind_mean',
        type=int,
        required=True,
        help='Wind mean',
    )
    args = parser.parse_args()
    args.plot = False
    return args

def simulate_and_save(args):
    config = gym_rl_mpc.SCENARIOS[args.env]['config'].copy()
    if args.psf:
        config['use_psf'] = True
        print("Using PSF corrected actions")
    config['wind_mean'] = args.wind_mean

    if not hasattr(args, 'save_sim_data'):
        args.save_sim_data = True

    env = gym.make(args.env, env_config=config)
    env_id = env.unwrapped.spec.id

    agent_path = args.agent
    agent = PPO.load(agent_path)
    sim_df = simulate_episode(env=env, agent=agent, max_time=args.time)
    if args.save_sim_data:
        agent_path_list = agent_path.split("\\")
        simdata_dir = os.path.join("logs", agent_path_list[-4], agent_path_list[-3], "sim_data")
        os.makedirs(simdata_dir, exist_ok=True)

        # Save file to logs\env_id\<EXPERIMENT_ID>\sim_data\<agent_file_name>_simdata.csv
        i = 0
        while os.path.exists(os.path.join(simdata_dir, env_id + "_" + agent_path_list[-1][0:-4] + f"_simdata_{i}.csv")):
            i += 1
        sim_df.to_csv(os.path.join(simdata_dir, env_id + "_" + agent_path_list[-1][0:-4] + f"_simdata_{i}.csv"))
    
    return sim_df, env

if __name__ == '__main__':
    args = parse_argument()

    if args.psf_agent:
        agent_paths_dict = agent_paths_psf
    else:
        agent_paths_dict = agent_paths

    for key in agent_paths_dict:
        agent_paths_dict[key] = agent_paths_dict[key][3:] + r"\agents\last_model_10000000.zip"

    num_episodes_per_agent = 1
    
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig2, (ax3, ax4) = plt.subplots(1, 2)
    agent_num = 0
    for agent_name, agent_path in agent_paths_dict.items():
        if '0' in agent_name or '5' in agent_name:
            print(agent_name)
            args.agent = agent_path
            color = 'C'+str(agent_num)
            for i in range(num_episodes_per_agent):
                sim_df, env = simulate_and_save(args)
                
                time = np.array(range(0, len(sim_df['theta']))) * env.step_size

                ax1.plot(time, sim_df['theta'] * RAD2DEG, color=color)
                ax1.set_ylabel('$\\theta$ [deg]')
                ax1.set_xlabel('Time [s]')
                ax1.set_aspect(args.time/10)
                ax1.set_yticks(np.arange(0,11,1))
                ax1.set_ylim([0,10])

                ax2.plot(time, sim_df['omega'] * RAD2RPM, color=color)
                ax2.set_ylabel('Rotor Velocity $\\Omega$ [RPM]')
                ax2.set_xlabel('Time [s]')
                
                ax2.set_aspect(aspect = args.time/7)
                ax2.set_yticks(np.arange(0,11,1))
                ax2.set_ylim([3,10])

                ax3.plot(time, sim_df['power'], color=color)
                ax3.set_ylabel('Power generated [W]')
                ax3.set_xlabel('Time [s]')
                ax3.set_ylim([-0.1e6,params.max_power_generation*1.05])
                ax3.set_ylim([0,1.1*15e6])

                ax4.plot(time, np.array(env.episode_history['wind_speed']), label='Wind speed')
                ax4.set_ylabel('Wind speed [m/s]')
                ax4.set_title('Wind')
            agent_num += 1

    custom_lines = [Line2D([0], [0], color='C0', lw=2),
                    Line2D([0], [0], color='C1', lw=2)]
    legends = ['Agent 0', 'Agent 5']
    ax1.legend(custom_lines, legends)
    ax2.legend(custom_lines, legends)
    ax3.legend(custom_lines, legends)
    fig.tight_layout()
    fig2.tight_layout()
    if args.save:
        if args.psf:
            fig.savefig(r'plot_scripts\plots\response_'+str(args.wind_mean)+'ms_psf.pdf', bbox_inches='tight')
        else:
            fig.savefig(r'plot_scripts\plots\response_'+str(args.wind_mean)+'ms.pdf', bbox_inches='tight')
    plt.show()


