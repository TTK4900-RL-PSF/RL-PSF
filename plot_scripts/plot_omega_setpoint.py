import argparse

import matplotlib.pyplot as plt
import numpy as np

import gym_rl_mpc.utils.model_params as params


def plot_omega_setpoint(save=False):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Make data.
    X = np.arange(0, 25, 0.001)
    Y = []
    for i in X:
        Y.append(params.omega_setpoint(i))
    Y = np.array(Y)*(60/(2*np.pi))

    # Plot the surface.
    reward_plot = ax.plot(X, Y)

    # Add axis labels
    ax.set_xlabel(r'Wind speed [m/s]')
    ax.set_ylabel(r'$\Omega_0$ [rpm]')
    ax.set_xlim([0,25])
    ax.set_ylim([0,10])
    ax.grid(True)
    ax.set_aspect(0.9)
    if save:
        plt.savefig('plots/omega_setpoint_curve.pdf', bbox_inches='tight')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--save',
        help='Save plots',
        action='store_true'
    )
    args = parser.parse_args()

    plot_omega_setpoint(save=args.save)
    plt.show()