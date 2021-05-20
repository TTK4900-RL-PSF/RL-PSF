import os

import numpy as np
from pandas import DataFrame

from gym_rl_mpc.utils.model_params import RAD2DEG


def format_history(env, lastn=-1):

    if lastn > -1:
        relevant_history = env.history[-min(lastn, len(env.history)):]
    else:
        relevant_history = env.history

    episode_nums = np.array([obj['episode_num'] for obj in relevant_history])
    crashes = np.array([obj['crashed'] for obj in relevant_history])
    no_crashes = crashes == 0
    crash_causes = np.array([obj['crash_cause'] for obj in relevant_history])
    avg_abs_theta = np.array([obj['avg_abs_theta'] for obj in relevant_history])
    avg_abs_theta_dot = np.array([obj['avg_abs_theta_dot'] for obj in relevant_history])
    std_theta = np.array([obj['std_theta'] for obj in relevant_history])
    rewards = np.array([obj['reward'] for obj in relevant_history])
    timesteps = np.array([obj['timesteps'] for obj in relevant_history])
    durations = np.array([obj['duration'] for obj in relevant_history])

    wind_speeds = np.array([obj['wind_speed'] for obj in relevant_history])
    theta_rewards = np.array([obj['theta_reward'] for obj in relevant_history])
    theta_dot_rewards = np.array([obj['theta_dot_reward'] for obj in relevant_history])
    omega_rewards = np.array([obj['omega_reward'] for obj in relevant_history])
    omega_dot_rewards = np.array([obj['omega_dot_reward'] for obj in relevant_history])
    power_rewards = np.array([obj['power_reward'] for obj in relevant_history])
    psf_rewards = np.array([obj['psf_reward'] for obj in relevant_history])
    psf_errors = np.array([obj['psf_error'] for obj in relevant_history])

    labels = np.array([r"episode", 
                        r"reward", 
                        r"crash", 
                        r"no_crash",
                        r"crash_cause",
                        r"theta",
                        r"theta_dot",
                        r"std_theta",
                        r"timesteps", 
                        r"duration", 
                        r"wind_speed", 
                        r"theta_reward", 
                        r"theta_dot_reward", 
                        r"omega_reward",
                        r"omega_dot_reward",
                        r"power_reward", 
                        r"psf_reward",
                        r"psf_error"])

    episode_nums = episode_nums.reshape((len(relevant_history), 1))
    rewards = rewards.reshape((len(relevant_history), 1))
    crashes = crashes.reshape((len(relevant_history), 1))
    no_crashes = no_crashes.reshape((len(relevant_history), 1))
    crash_causes = crash_causes.reshape((len(relevant_history), 1))
    avg_abs_theta = avg_abs_theta.reshape((len(relevant_history), 1))
    avg_abs_theta_dot = avg_abs_theta_dot.reshape((len(relevant_history), 1))
    std_theta = std_theta.reshape((len(relevant_history), 1))
    timesteps = timesteps.reshape((len(relevant_history), 1))
    durations = durations.reshape((len(relevant_history), 1))

    wind_speeds = wind_speeds.reshape((len(relevant_history), 1))
    theta_rewards = theta_rewards.reshape((len(relevant_history), 1))
    theta_dot_rewards = theta_dot_rewards.reshape((len(relevant_history), 1))
    omega_rewards = omega_rewards.reshape((len(relevant_history), 1))
    omega_dot_rewards = omega_dot_rewards.reshape((len(relevant_history), 1))
    power_rewards = power_rewards.reshape((len(relevant_history), 1))
    psf_rewards = psf_rewards.reshape((len(relevant_history), 1))
    psf_errors = psf_errors.reshape((len(relevant_history), 1))

    report_data = np.hstack([   episode_nums,
                                rewards,
                                crashes,
                                no_crashes,
                                crash_causes,
                                avg_abs_theta,
                                avg_abs_theta_dot,
                                std_theta,
                                timesteps,
                                durations,
                                wind_speeds,
                                theta_rewards,
                                theta_dot_rewards,
                                omega_rewards,
                                omega_dot_rewards,
                                power_rewards,
                                psf_rewards,
                                psf_errors,
                            ])

    df = DataFrame(report_data, columns=labels)

    return df

def report(env, report_dir):
    try:
        os.makedirs(report_dir, exist_ok=True)
        df = format_history(env)

        file_path = os.path.join(report_dir, "history_data.csv")
        if not os.path.isfile(file_path):
            df.to_csv(file_path)
        else:
            df.to_csv(file_path, mode='a', header=False)
        # print("reported episode to file")
    except PermissionError as e:
        print('Warning: Report files are open - could not update report: ' + str(repr(e)))
    except OSError as e:
        print('Warning: Ignoring OSError: ' + str(repr(e)))

def make_summary_file(data, report_dir, total_num_episodes):
    os.makedirs(report_dir, exist_ok=True)

    crashes = np.array(data['crash'])
    no_crashes = crashes == 0
    avg_abs_theta = np.array(data['theta'])
    std_theta = np.array(data['std_theta'])
    avg_abs_theta_dot = np.array(data['theta_dot'])
    rewards = np.array(data['reward'])
    timesteps = np.array(data['timesteps'])
    durations = np.array(data['duration'])

    with open(os.path.join(report_dir, 'summary.txt'), 'w') as f:
        f.write('# TOTAL EPISODES TRAINED: {}\n'.format(total_num_episodes))
        f.write('# PERFORMANCE METRICS (LAST {} EPISODES AVG.)\n'.format(data.shape[0]))
        f.write('{:<30}{:<30}\n'.format('Avg. Reward', rewards.mean()))
        f.write('{:<30}{:<30}\n'.format('Std. Reward', rewards.std()))
        f.write('{:<30}{:<30.2%}\n'.format('Avg. Crashes', crashes.mean()))
        f.write('{:<30}{:<30.2%}\n'.format('No Crashes', no_crashes.mean()))
        f.write('{:<30}{:<30}\n'.format('Avg. Absolute theta [deg]', avg_abs_theta.mean()*RAD2DEG))
        f.write('{:<30}{:<30}\n'.format('Avg. Std. theta [deg]', std_theta.mean()*RAD2DEG))
        f.write('{:<30}{:<30}\n'.format('Avg. Absolute theta_dot [deg/s]', avg_abs_theta_dot.mean()*RAD2DEG))
        f.write('{:<30}{:<30}\n'.format('Avg. Timesteps', timesteps.mean()))
        f.write('{:<30}{:<30}\n'.format('Avg. Duration', durations.mean()))
