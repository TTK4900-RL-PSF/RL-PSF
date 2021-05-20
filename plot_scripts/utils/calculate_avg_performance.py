import argparse
import numpy as np
import pandas as pd
import argparse



def calculate_avg_performance(filename):
    df = pd.read_csv(filename)
    crashes = df['crash'].values
    performances = df['performance'].values
    psf_rewards = df['cumulative_psf_reward'].values
    crash_indexes = []
    for i in range(len(crashes)):
        if crashes[i]:
            crash_indexes.append(i)
    performances_no_crash = np.delete(performances,crash_indexes)
    psf_rewards_no_crash = np.delete(psf_rewards,crash_indexes)
    avg_perf = np.sum(performances_no_crash)/(len(crashes)-np.sum(crashes))
    avg_crash = np.sum(crashes)*100/len(crashes)
    avg_psf_reward = np.sum(psf_rewards_no_crash)/(len(crashes)-np.sum(crashes))
    return avg_perf, avg_crash, avg_psf_reward

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--file',
        type=str,
        required=True,
        help="Filename"
    )
    args = parser.parse_args()
    avg_performance, avg_crashes, avg_psf_reward = calculate_avg_performance(args.file)
    print(f'Avg. Performance: {avg_performance}')
    print(f'Crashes: {avg_crashes}%')