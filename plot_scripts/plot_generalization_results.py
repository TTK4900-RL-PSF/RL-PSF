import argparse
from plot_scripts.utils.calculate_avg_performance import calculate_avg_performance
from plot_scripts.utils.agent_paths import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import os

font_dict=dict(family='Arial',
            size=18,
            color='black'
            )

def get_data(paths_dict, psf_test_arg):

    performance_data = []
    crash_data = []
    psf_reward_data = []
    for _, path in paths_dict.items():
        agent_folder = os.path.join(path,'test_data')
        agent_performances = []
        agent_crashes = []
        agent_psf_rewards = []
        for filename in os.listdir(agent_folder):
            if filename.endswith(".csv") and '6000' in filename and 'PSFtest' not in filename:
                if (psf_test_arg and '_PSF_' in filename) or (not psf_test_arg and '_PSF_' not in filename):
                    file = os.path.join(agent_folder, filename)
                    perf, crash, psf_reward = calculate_avg_performance(file)
                    agent_performances.append(perf)
                    agent_crashes.append(crash)
                    agent_psf_rewards.append(psf_reward)
        performance_data.append(agent_performances)
        crash_data.append(agent_crashes)
        psf_reward_data.append(agent_psf_rewards)
    return np.array(performance_data), np.array(crash_data), np.array(psf_reward_data)


def plot_gen_heatmap(performance_data, crash_data, psf_reward_data, save=False):
    textcolors=["white","black"]

    # data = psf_reward_data
    # ylabel = "$\\frac{PSF reward}{Min. PSF reward}$"
    # filename = "plots/generalization_psf_reward_heatmap_6000_psf.pdf"
    # format = mtick.PercentFormatter(decimals=2)
    # cmap = 'autumn_r'
    # textcolors.reverse()
    # limits = [0,0.5]

    data = performance_data
    ylabel = "Performance"
    filename = "plots/generalization_performance_heatmap_6000_noPSFagent_PSFtest.pdf"
    format = "{:.2f}".format
    cmap = 'autumn'
    limits = [54,70]
    
    # data = crash_data
    # ylabel = "Crash rate"
    # filename = "plots/generalization_crash_heatmap_6000_noPSFagent_PSFtest.pdf"
    # format = mtick.PercentFormatter(decimals=0)
    # cmap = 'autumn_r'
    # textcolors.reverse()
    # limits = [0, 45]
    
    labels = ["Level 0", "Level 1", "Level 2", "Level 3", "Level 4", "Level 5", "Level HighWinds"]
    labels = labels[:len(data)]

    fig, ax = plt.subplots()
    im = ax.imshow(data, cmap=cmap, clim=limits)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, format=format)
    cbar.axt.set_ylabel(ylabel, rotation=-90, va="bottom")
    # We want to show all ticks...
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_ylabel("Agent Train Level")
    ax.set_xlabel("Test Level")

    # Normalize the threshold to the images color range.
    threshold = limits[0]+(limits[1]-limits[0])/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if False:#i==j:
                ax.text(j, i, '-',ha="center", va="center", color='black')
            else:
                ax.text(j, i, format(data[i, j]),ha="center", va="center", color=textcolors[int(data[i, j] > threshold)])
                

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")

    fig.tight_layout()
    if save:
        plt.savefig(filename, bbox_inches='tight')
    plt.show()

def plot_training_performance(performance_data, crash_data, save=False):

    labels = ["Level 0", "Level 1", "Level 2", "Level 3", "Level 4", "Level 5"]
    fig, ax = plt.subplots()
    new_performance_data = [performance_data[i][i] for i in range(len(performance_data[0]))]
    new_crash_data = [crash_data[i][i] for i in range(len(crash_data[0]))]

    lns1 = ax.bar(labels, new_performance_data, width=0.5, label='Performance')
    ax.set_ylabel('Performance')
    ax.set_ylim([0,100])
    ax.set_axisbelow(True)
    ax.grid(True, color='gray', axis='y', linestyle='dashed')
    ax.set_yticks(np.arange(0,101,5))
    # ax.set_ylim([0,100])
    # ax.tick_params(axis='y', colors='darkorange')

    ax2 = ax.twinx()
    lns2 = ax2.bar(labels, new_crash_data, width=0.5, color="red", label='Crash Rate')
    ax2.set_ylabel('Crash rate')
    ax2.set_ylim([0,100])
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    ax2.tick_params(axis='y', colors='red')
    ax2.set_yticks(np.arange(0,101,5))
    
    # ask matplotlib for the plotted objects and their labels
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc=0)

    fig.tight_layout()
    if save:
        if args.psf:
            plt.savefig("plots/training_performance_bar_6000_psf.pdf", bbox_inches='tight')
        else:
            plt.savefig("plots/training_performance_bar_6000.pdf", bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
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
        '--psf_test',
        help='Plot results with PSF on during testing',
        action='store_true'
    )
    args = parser.parse_args()

    if args.psf_agent:
        performance_data, crash_data, psf_reward_data = get_data(agent_paths_psf, args.psf_test)
    else:
        performance_data, crash_data, psf_reward_data = get_data(agent_paths, args.psf_test)
    
    performance_data = 100*performance_data/(3*6000)
    psf_reward_data = 100*psf_reward_data/(-1*5*4.2*6000)

    plot_gen_heatmap(performance_data, crash_data, psf_reward_data, save=args.save)
    # plot_training_performance(performance_data, crash_data, save=args.save)