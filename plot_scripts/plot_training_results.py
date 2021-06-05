import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.ticker as mtick
import scipy.interpolate as interp

def smooth(scalars, weight):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value

    return smoothed

def plot_ep_rew_mean(filepaths, labels=None, save=False):
    max_timesteps = 0
    fig = plt.figure()
    ax = fig.add_subplot(111)

    for i in range(len(filepaths)):
        df = pd.read_csv(filepaths[i])
        X = df['Step']/1e6
        Y = df['Value']
        if labels:
            if "with psf" in labels[i].lower():
                ax.plot(X, Y, label=labels[i], linestyle='dashed')#, color="C"+str(i-1))
            else:
                ax.plot(X, Y, label=labels[i])
        else:
            ax.plot(X, Y)
        max_timesteps = np.maximum(max_timesteps, np.max(X))

    # Add axis labels
    ax.set_xlabel(r'Timestep (in million)')
    ax.set_ylabel(r'Episode Reward Mean')
    ax.set_xlim([0,max_timesteps])
    ax.grid(True)
    
    if labels:
        ax.legend(loc='lower right')

    if save:
        plt.savefig('plots/ep_rew_mean_noPSF_vs_PSF_10M.pdf', bbox_inches='tight')
    
    plt.show()

def plot_ep_crash_mean(filepaths, labels=None, save=False):
    max_timesteps = 0

    fig = plt.figure()
    ax = fig.add_subplot(111)

    for i in range(len(filepaths)):
        df = pd.read_csv(filepaths[i])
        X = df['Step']/1e6
        Y = df['Value']*100
        Y = smooth(Y,0.89)
        if labels:
            if "with psf" in labels[i].lower():
                if i>1:
                    ax.plot(X, Y, label=labels[i], linestyle='dashed', dashes=(4, 3))
                else:
                    ax.plot(X, Y, label=labels[i], linestyle='dashed')
            else:
                ax.plot(X, Y, label=labels[i])
        else:
            ax.plot(X, Y, zorder=i)
        max_timesteps = np.maximum(max_timesteps, np.max(X))

    # Add axis labels
    ax.set_xlabel(r'Timestep (in million)')
    ax.set_ylabel(r'Crash rate mean')
    ax.set_xlim([0,max_timesteps])
    ax.grid(True)
    ax.set_ylim([-5,100])

    formater = mtick.PercentFormatter(decimals=0)
    ax.yaxis.set_major_formatter(formater)
    
    if labels:
        ax.legend()

    if save:
        plt.savefig('plots/ep_crash_mean_tight_constraints.pdf', bbox_inches='tight')
    
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--file',
        nargs='+',
        help='Path to the CSV file.',
        type=str,
    )
    parser.add_argument(
        '--label',
        nargs='+',
        help='Labels for each file',
        type=str,
    )
    parser.add_argument(
        '--save',
        help='Save plots',
        action='store_true'
    )
    args = parser.parse_args()

    if args.label:
        assert len(args.file)==len(args.label)

    if args.file:
        filepaths = args.file
    else:
        filepaths = [   r"C:\Users\halvorot\Downloads\run-VariableWindLevel0-v17_1618915245ppo_tensorboard_PPO_1-tag-rollout_ep_rew_mean.csv",
                        r"C:\Users\halvorot\Downloads\run-VariableWindLevel1-v17_1618921752ppo_tensorboard_PPO_1-tag-rollout_ep_rew_mean.csv",
                        r"C:\Users\halvorot\Downloads\run-VariableWindLevel2-v17_1618928488ppo_tensorboard_PPO_1-tag-rollout_ep_rew_mean.csv",
                        r"C:\Users\halvorot\Downloads\run-VariableWindLevel3-v17_1618934307ppo_tensorboard_PPO_1-tag-rollout_ep_rew_mean.csv",
                        r"C:\Users\halvorot\Downloads\run-VariableWindLevel4-v17_1618940658ppo_tensorboard_PPO_1-tag-rollout_ep_rew_mean.csv",
                        r"C:\Users\halvorot\Downloads\run-VariableWindLevel5-v17_1618946704ppo_tensorboard_PPO_1-tag-rollout_ep_rew_mean.csv",
                        r"C:\Users\halvorot\Downloads\run-VariableWindPSFtest-v17_1619770390ppo_tensorboard_PPO_1-tag-rollout_ep_rew_mean (1).csv",
                        r"C:\Users\halvorot\Downloads\run-VariableWindLevel0-v17_1619804074ppo_tensorboard_PPO_1-tag-rollout_ep_rew_mean.csv", 
                        r"C:\Users\halvorot\Downloads\run-VariableWindLevel1-v17_1619817419ppo_tensorboard_PPO_1-tag-rollout_ep_rew_mean.csv", 
                        r"C:\Users\halvorot\Downloads\run-VariableWindLevel2-v17_1619805498ppo_tensorboard_PPO_1-tag-rollout_ep_rew_mean.csv", 
                        r"C:\Users\halvorot\Downloads\run-VariableWindLevel3-v17_1619826109ppo_tensorboard_PPO_1-tag-rollout_ep_rew_mean.csv", 
                        r"C:\Users\halvorot\Downloads\run-VariableWindLevel4-v17_1619817419ppo_tensorboard_PPO_1-tag-rollout_ep_rew_mean.csv", 
                        r"C:\Users\halvorot\Downloads\run-VariableWindLevel5-v17_1619809322ppo_tensorboard_PPO_1-tag-rollout_ep_rew_mean.csv",
                        r"C:\Users\halvorot\Downloads\run-VariableWindPSFtest-v17_1619696400ppo_tensorboard_PPO_1-tag-rollout_ep_rew_mean (2).csv"
                    ]
        filepaths = [filepaths[0],filepaths[7], filepaths[6], filepaths[13]]
        filepaths = [   r"C:\Users\halvorot\Downloads\run-VariableWindLevel0-v17_1619804074ppo_tensorboard_PPO_1-tag-custom_crashed (2).csv",
                        r"C:\Users\halvorot\Downloads\run-VariableWindLevel1-v17_1619817419ppo_tensorboard_PPO_1-tag-custom_crashed (1).csv",
                        r"C:\Users\halvorot\Downloads\run-VariableWindLevel2-v17_1619805498ppo_tensorboard_PPO_1-tag-custom_crashed (1).csv",
                        r"C:\Users\halvorot\Downloads\run-VariableWindLevel3-v17_1619826109ppo_tensorboard_PPO_1-tag-custom_crashed (1).csv",
                        r"C:\Users\halvorot\Downloads\run-VariableWindLevel4-v17_1619817419ppo_tensorboard_PPO_1-tag-custom_crashed (1).csv",
                        r"C:\Users\halvorot\Downloads\run-VariableWindLevel5-v17_1619809322ppo_tensorboard_PPO_1-tag-custom_crashed (1).csv",
                    ]
        filepaths = [   r"C:\Users\halvorot\Downloads\run-ConstantWindLevel1-v17_1620385846ppo_tensorboard_PPO_1-tag-custom_crashed (1).csv",
                        r"C:\Users\halvorot\Downloads\run-ConstantWindLevel1-v17_1620385841ppo_tensorboard_PPO_1-tag-custom_crashed (1).csv",
                        r"C:\Users\halvorot\Downloads\run-ConstantWindLevel2-v17_1620388963ppo_tensorboard_PPO_1-tag-custom_crashed (1).csv",                        
                        r"C:\Users\halvorot\Downloads\run-ConstantWindLevel2-v17_1620385841ppo_tensorboard_PPO_1-tag-custom_crashed (1).csv",
                    ]
        filepaths = [   r"C:\Users\halvorot\Downloads\run-VariableWindLevel0-v17_1618915245ppo_tensorboard_PPO_1-tag-custom_crashed (3).csv",
                        r"C:\Users\halvorot\Downloads\run-VariableWindLevel0-v17_1619804074ppo_tensorboard_PPO_1-tag-custom_crashed (4).csv",
                        r"C:\Users\halvorot\Downloads\run-VariableWindPSFtest-v17_1619770390ppo_tensorboard_PPO_1-tag-custom_crashed (1).csv",                        
                        r"C:\Users\halvorot\Downloads\run-VariableWindPSFtest-v17_1619696400ppo_tensorboard_PPO_1-tag-custom_crashed (2).csv",
                    ]
    if args.label:
        labels = args.label
    else:
        labels = [  "Level 0",
                    "Level 1",
                    "Level 2",
                    "Level 3",
                    "Level 4",
                    "Level 5",
                    "Level HighWinds",
                    "Level 0 with PSF",
                    "Level 1 with PSF",
                    "Level 2 with PSF",
                    "Level 3 with PSF",
                    "Level 4 with PSF",
                    "Level 5 with PSF",
                    "Level HighWinds with PSF"
                    ]
        labels = [labels[0], labels[7], labels[6], labels[13]]
        labels = [  "Level 0",
                    "Level 1",
                    "Level 2",
                    "Level 3",
                    "Level 4",
                    "Level 5"
        ]
        labels = [  "Level ConstantLow",
                    "Level ConstantLow with PSF",
                    "Level ConstantHigh",
                    "Level ConstantHigh with PSF"
        ]
        labels = [  "Level 0",
                    "Level 0 with PSF",
                    "Level HighWinds",
                    "Level HighWinds with PSF"
        ]
        labels=None
    
    plot_ep_crash_mean(filepaths, labels, save=args.save)