import numpy as np


def ssa(angle):
    return ((angle + np.pi) % (2*np.pi)) - np.pi
