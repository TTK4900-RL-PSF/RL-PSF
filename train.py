import gym_rl_mpc
import argparse
import json
import multiprocessing
import os
from time import time

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

import gym_rl_mpc
from gym_rl_mpc import reporting

def linear_schedule(initial_value):
    """
    Linear learning rate schedule.
    :param initial_value: (float or str)
    :return: (function)
    """
    if isinstance(initial_value, str):
        initial_value = float(initial_value)

    def func(progress):
        """
        Progress will decrease from 1 (beginning) to 0
        :param progress: (float)
        :return: (float)
        """
        return progress * initial_value

    return func

hyperparams = {
    'n_steps': 1024,
    'learning_rate': linear_schedule(initial_value=1e-4),
    'gae_lambda': 0.95,
    'gamma': 0.99,
    'n_epochs': 4,
    'clip_range': 0.2,
    'ent_coef': 0.01,
}

class ReportingCallback(BaseCallback):
    """
    Callback for reporting training
    :param report_dir: Path to the folder where the report will be saved.
    :param verbose:
    """

    def __init__(self, report_dir: str, verbose: int = 0):
        super(ReportingCallback, self).__init__(verbose)
        self.report_dir = report_dir
        self.verbose = verbose

    def _on_step(self) -> bool:
        # check if env is done, if yes report it to csv file
        done_array = np.array(self.locals.get("done") if self.locals.get("done") is not None else self.locals.get("dones"))
        if np.sum(done_array).item() > 0:
            env_histories = self.training_env.get_attr('history')

            class Struct(object): pass
            report_env = Struct()
            report_env.history = []
            for env_idx in range(len(done_array)):
                if done_array[env_idx]:
                    report_env.history.append(env_histories[env_idx])

            reporting.report(env=report_env, report_dir=self.report_dir)


        return True

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        vec_env = self.training_env
        env_histories = vec_env.get_attr('total_history')
        class Struct(object): pass
        report_env = Struct()
        report_env.history = []
        for episode in range(max(map(len, env_histories))):
            for env_idx in range(len(env_histories)):
                if (episode < len(env_histories[env_idx])):
                    report_env.history.append(env_histories[env_idx][episode])

        if len(report_env.history) > 0:
            training_data = reporting.format_history(report_env, lastn=100)
            reporting.make_summary_file(training_data, self.report_dir, len(report_env.history))
            if self.verbose:
                print("Made summary file of training")

            total_history_data = reporting.format_history(report_env)
            os.makedirs(self.report_dir, exist_ok=True)
            file_path = os.path.join(self.report_dir, "total_history_data.csv")
            total_history_data.to_csv(file_path)
            if self.verbose:
                print("Made total history file of training")

class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        self.start_time = time()
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        done_array = np.array(self.locals.get("done") if self.locals.get("done") is not None else self.locals.get("dones"))

        if np.sum(done_array).item():
            history = self.training_env.get_attr('history')

            for env_idx in range(len(done_array)):
                if done_array[env_idx]:
                    self.logger.record_mean('custom/crashed', history[env_idx]['crashed'])
                    self.logger.record_mean('custom/psf_error', history[env_idx]['psf_error'])
                    self.logger.record_mean('custom/wind_speed', history[env_idx]['wind_speed'])
                    self.logger.record_mean('custom/theta_reward', history[env_idx]['theta_reward'])
                    self.logger.record_mean('custom/theta_dot_reward', history[env_idx]['theta_dot_reward'])
                    self.logger.record_mean('custom/omega_reward', history[env_idx]['omega_reward'])
                    self.logger.record_mean('custom/omega_dot_reward', history[env_idx]['omega_dot_reward'])
                    self.logger.record_mean('custom/power_reward', history[env_idx]['power_reward'])
                    self.logger.record_mean('custom/psf_reward', history[env_idx]['psf_reward'])

        self.logger.record("time/custom_time_elapsed", int(time() - self.start_time))
        episodesList = np.array(self.training_env.get_attr('episode'))
        num_episodes = np.sum(episodesList)
        self.logger.record("time/num_episodes", num_episodes)

        return True


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--timesteps',
        type=int,
        default=2000000,
        help='Number of timesteps to train the agent. Default=2000000',
    )
    parser.add_argument(
        '--agent',
        help='Path to the RL agent to continue training from.',
    )
    parser.add_argument(
        '--env',
        type=str,
        required=True,
        choices=gym_rl_mpc.SCENARIOS.keys(),
        help="Environment to run."
    )
    parser.add_argument(
        '--note',
        type=str,
        default=None,
        help="Note with additional info about training"
    )
    parser.add_argument(
        '--no_reporting',
        help='Skip reporting to increase framerate',
        action='store_true'
    )
    parser.add_argument(
        '--psf',
        help='Use psf corrected action',
        action='store_true'
    )
    parser.add_argument(
        '--num_cpus',
        type=int,
        help='Manually set number of CPUs to use'
    )
    parser.add_argument(
        '--psf_T',
        type=int,
        default=None,
        help='psf horizon'
    )
    parser.add_argument(
        '--psf_lb_omega',
        type=float,
        default=None,
        help='omega lower bound'
    )
    parser.add_argument(
        '--psf_ub_omega',
        type=float,
        default=None,
        help='upper bound omega'
    )
    args = parser.parse_args()

    NUM_CPUs = multiprocessing.cpu_count() if not args.num_cpus else args.num_cpus

    # Make environment (NUM_CPUs parallel envs)
    env_id = args.env
    customconfig = gym_rl_mpc.SCENARIOS[args.env]['config'].copy()
    if args.psf_lb_omega:
        customconfig['psf_lb_omega'] = args.psf_lb_omega
    if args.psf_ub_omega:
        customconfig['psf_ub_omega'] = args.psf_ub_omega
    if args.psf_T:
        customconfig['psf_T'] = args.psf_T
    if args.psf:
        customconfig['use_psf'] = True
        print("Using PSF corrected actions")

    env_kwargs = {'env_config': customconfig}

    env = make_vec_env(env_id, n_envs=NUM_CPUs, vec_env_cls=SubprocVecEnv, env_kwargs=env_kwargs)


    # Define necessary directories
    EXPERIMENT_ID = str(int(time())) + 'ppo'
    agents_dir = os.path.join('logs', env_id, EXPERIMENT_ID, 'agents')
    os.makedirs(agents_dir, exist_ok=True)
    report_dir = os.path.join('logs', env_id, EXPERIMENT_ID, 'training_report')
    tensorboard_log = os.path.join('logs', env_id, EXPERIMENT_ID, 'tensorboard')

    # Write note and config to Note.txt file
    with open(os.path.join('logs', env_id, EXPERIMENT_ID, "Note.txt"), "a") as file_object:
        hyperparams_edit = hyperparams.copy()
        if hyperparams['learning_rate']:
            hyperparams_edit['learning_rate'] = str(hyperparams_edit['learning_rate'])
        file_object.write("env_config: " + json.dumps(env.get_attr('config')[0]) + "\n")
        file_object.write("hyperparams: " + json.dumps(hyperparams_edit) + "\n")
        if args.agent:
            file_object.write(f"Continued training from: {args.agent}\n")
        if args.note:
            file_object.write(args.note)

    # Callback to save model at checkpoints during training
    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=agents_dir)
    # Callback to report training to file
    reporting_callback = ReportingCallback(report_dir=report_dir, verbose=True)
    # Callback to report additional values to tensorboard
    tensorboard_callback = TensorboardCallback(verbose=True)
    # Create the callback list
    if args.no_reporting:
        callback = CallbackList([checkpoint_callback, tensorboard_callback])
    else:
        callback = CallbackList([checkpoint_callback, reporting_callback, tensorboard_callback])

    if (args.agent is not None):
        agent = PPO.load(args.agent, env=env, verbose=True, tensorboard_log=tensorboard_log)
    else:
        agent = PPO('MlpPolicy', env, verbose=True, tensorboard_log=tensorboard_log, **hyperparams)

    agent.learn(total_timesteps=args.timesteps, callback=callback)

    # Save trained agent
    agent_path = os.path.join(agents_dir, "last_model_" + str(args.timesteps))
    agent.save(agent_path)

    env.close()
