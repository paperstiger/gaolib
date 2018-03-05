#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Gao Tang <gt70@duke.edu>
#
# Distributed under terms of the MIT license.

"""
common.py

Define global variables being shared across this module
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


plotkwargs = ['alpha', 'color', 'c', 'label', 'linestyle', 'ls', 'linewidth', 'lw', 'marker']
scatterkwargs = ['color', 's', 'marker', 'c', 'cmap', 'norm', 'vmin', 'vmax', 'alpha']


def getColorCycle():
    return plt.rcParams['axes.prop_cycle'].by_key()['color']


def get3dAxis():
    """Return a fig and an axis that supports 3d plotting"""
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    return fig, ax


def getIndAlongAxis(x, axis, ind=0):
    if x.ndim == 1:
        return x
    slc = [slice(None)] * len(x.shape)
    slc[axis] = ind
    return x[slc]
