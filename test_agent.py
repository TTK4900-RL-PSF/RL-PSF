import argparse
import os
from pathlib import Path

import gym
import numpy as np
from pandas.core.frame import DataFrame
from stable_baselines3 import PPO

import gym_rl_mpc
import utils
from gym_rl_mpc.utils import model_params as params
from gym_rl_mpc.utils.model_params import RAD2DEG, RAD2RPM


def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--env',
        type=str,
        required=True,
        choices=gym_rl_mpc.SCENARIOS.keys(),
        help="Environment to run."
    )
    parser.add_argument(
        '--agent',
        required=True,
        help='Path to agent .zip file.',
    )
    parser.add_argument(
        '--num_episodes',
        type=int,
        required=True,
        default=100,
        help='Number of episodes to simulate.',
    )
    parser.add_argument(
        '--time',
        type=int,
        default=300,
        help='Max simulation time (seconds).',
    )
    parser.add_argument(
        '--psf',
        help='Use psf corrected action',
        action='store_true'
    )
    args = parser.parse_args()
    return args


def test(args):
    if isinstance(args, dict):
        args = argparse.Namespace(**args)
    
    # Change config if changes defined
    config = gym_rl_mpc.SCENARIOS[args.env]['config'].copy()
    config['max_episode_time'] = args.time
    if args.psf:
        config['use_psf'] = True
        print("Using PSF corrected actions")
    
    # Create environment
    env = gym.make(args.env, env_config=config)
    env_id = env.unwrapped.spec.id

    # Load agent
    agent_path = Path(args.agent)
    agent = PPO.load(agent_path)

    # Simulate episodes and save results to lists
    cumulative_rewards = []
    cumulative_psf_rewards = []
    crashes = []
    print(f'Testing in "{env_id}" for {args.num_episodes} episodes.\n ')
    report_msg_header = '{:<20}{:<20}{:<20}{:<20}'.format('Episode', 'Timesteps', 'Cum. Reward', 'Progress')
    print(report_msg_header)
    print('-'*len(report_msg_header)) 
    for episode in range(args.num_episodes):
        sim_df = utils.simulate_episode(env=env, agent=agent, max_time=args.time, verbose=True, id=f'Episode {episode}')
        cumulative_rewards.append(np.sum(sim_df['reward']))
        cumulative_psf_rewards.append(np.sum(sim_df['psf_reward']))
        crashes.append(env.crashed)
        print('')
    performances = np.subtract(cumulative_rewards, cumulative_psf_rewards)
    crash_indexes = []
    for i in range(len(crashes)):
        if crashes[i]:
            crash_indexes.append(i)
    performances_no_crash = np.delete(performances, crash_indexes)
    print(f'Avg. Cumulative reward: {np.sum(cumulative_rewards)/args.num_episodes}')
    if args.num_episodes-np.sum(crashes) != 0:
        print(f'Avg. Performance: {np.sum(performances_no_crash)/(args.num_episodes-np.sum(crashes))}')
    print(f'Crashes: {np.sum(crashes)*100/args.num_episodes}%')

    # Make dataframe to save results to file
    test_df = DataFrame(list(zip(cumulative_rewards, cumulative_psf_rewards, performances, crashes)), columns=['cumulative_reward','cumulative_psf_reward','performance','crash'])
    
    # Save results to file
    agent_path_list = agent_path.parts
    testdata_dir = Path("logs", agent_path_list[-4], agent_path_list[-3], "test_data")
    os.makedirs(testdata_dir, exist_ok=True)
    if args.psf:
        psf_prefix = "_PSF_"
    else:
        psf_prefix = "_"
    i = 0
    while os.path.exists(os.path.join(testdata_dir, env_id + "_" + agent_path_list[-1][0:-4] + psf_prefix + str(args.time*10) + f"_testdata_{i}.csv")):
        i += 1
    test_df.to_csv(os.path.join(testdata_dir, env_id + "_" + agent_path_list[-1][0:-4] + psf_prefix + str(args.time*10) + f"_testdata_{i}.csv"))
    print(f'Reported to file in: {testdata_dir}')
    env.close()


if __name__ == "__main__":
    args = parse_argument()
    test(args)
