"""
A generator for an arbitrary number of `for` loops.

From Python Recipes.
"""


import numpy as np


__all__ = ['mrange']


def mrange(minvec, maxvec=None):
    if maxvec is None:
        maxvec = minvec
        minvec = [0] * len(maxvec)
    vec = list(minvec)
    unitpos = len(vec) - 1
    maxunit = maxvec[unitpos]
    _tuple = tuple
    while 1:
        if vec[unitpos] == maxunit:
            i = unitpos
            while vec[i] == maxvec[i]:
                vec[i] = minvec[i]
                i -= 1
                if i == -1:
                    return
                vec[i] += 1
        yield np.array(vec)
        vec[unitpos] += 1
