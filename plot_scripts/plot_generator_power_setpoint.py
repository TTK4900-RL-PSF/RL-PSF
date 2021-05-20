import argparse

import matplotlib.pyplot as plt
import numpy as np

import gym_rl_mpc.utils.model_params as params


def plot_gen_pow(save=False):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Make data.
    X = np.arange(0, 25, 0.001)
    Y = []
    for i in X:
        Y.append(params.power_regime(i))
    Y = np.array(Y)/1e6

    # Plot the surface.
    reward_plot = ax.plot(X, Y)

    # Add axis labels
    ax.set_xlabel(r'Wind speed [m/s]')
    ax.set_ylabel(r'Generator Power [MW]')
    ax.set_xlim([0,25])
    ax.set_ylim([-0.1,20])
    ax.grid(True)
    ax.set_aspect(0.4)
    if save:
        plt.savefig('plots/generator_power_setpoint_curve.pdf', bbox_inches='tight')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--save',
        help='Save plots',
        action='store_true'
    )
    args = parser.parse_args()

    plot_gen_pow(save=args.save)
    plt.show()