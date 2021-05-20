import argparse

import matplotlib.pyplot as plt
import numpy as np


def plot_pow_gen(save=False):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Make data.
    X = np.arange(2000, 2031, 1)
    historical = np.array([0.1,0.2,0.4,1.3,2.0,2.5,3.1,4.1,5.4,5.0,7.7,11.7,14.8,20.7,24.6,39.4,41.2,55.9,66.5,0,0,0,0,0,0,0,0,0,0,0,0])
    predicted = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,307.5,0,0,0,0,606.3])

    # Plot the surface.
    ax.bar(X, historical, label='Historical')
    ax.bar(X, predicted, label='SDS')

    # Add axis labels
    ax.set_xlabel(r'Year')
    ax.set_ylabel(r'Offshore wind power generation [TWh]')
    ax.set_aspect(0.03)
    ax.legend()
    if save:
        plt.savefig('plots/offshore_power_generation_SDS.pdf', bbox_inches='tight')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--save',
        help='Save plots',
        action='store_true'
    )
    args = parser.parse_args()

    plot_pow_gen(save=args.save)
    plt.show()