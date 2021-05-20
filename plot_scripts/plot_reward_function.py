import argparse

import matplotlib.pyplot as plt
import numpy as np


def r_theta():
    gamma_theta = 0.12

    X = np.arange(-10, 10, 0.001)
    Y = np.exp(-gamma_theta*(np.abs(X))) - gamma_theta*np.abs(X)
    return X, Y 

def r_theta_dot():
    reward_theta_dot = 3

    X = np.arange(-2.5, 2.5, 0.001)
    Y = -reward_theta_dot*X**2
    return X, Y 
    
def r_omega():
    gamma_omega = 0.285

    X = np.arange(-10, 10, 0.001)
    Y = np.exp(-gamma_omega*(np.abs(X))) - gamma_omega*np.abs(X)
    return X, Y

def r_omega_dot():
    reward_omega_dot = 4

    X = np.arange(-2.5, 2.5, 0.001)
    Y = -reward_omega_dot*X**2
    return X, Y 

def r_power():
    gamma_power = 0.1

    X = np.arange(-15, 15, 0.001)
    Y = np.exp(-gamma_power*np.abs(X)) - gamma_power*np.abs(X)
    return X, Y 
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--save',
        help='Save plots',
        action='store_true'
    )
    args = parser.parse_args()

    fig = plt.figure()
    ax = fig.add_subplot(111)

    x, y = r_omega_dot()

    ax.plot(x, y)

    # Add axis labels
    ax.set_xlabel(r'$\dot{\Omega}$ $[\frac{RPM}{s}]$')
    ax.set_ylabel(r'Reward')
    ax.set_xlim([-1.1,1.1])
    ax.set_ylim([-1.1,0.1])
    ax.grid(True)

    if args.save:
        plt.savefig('plots/r_omega_dot.pdf', bbox_inches='tight')

    plt.show()