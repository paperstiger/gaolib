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
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from itertools import chain


plotkwargs = ['alpha', 'color', 'c', 'label', 'linestyle', 'ls', 'linewidth', 'lw', 'marker']
scatterkwargs = ['color', 's', 'marker', 'c', 'cmap', 'norm', 'vmin', 'vmax', 'alpha']


def getColorCycle():
    return plt.rcParams['axes.prop_cycle'].by_key()['color']


def get3dAxis():
    """Return a fig and an axis that supports 3d plotting"""
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    return fig, ax


def subplots3d(nrow=None, ncol=None, **kwargs):
    """Create subplots."""
    return subplots(nrow, ncol, subplot_kw={'projection': '3d'}, **kwargs)


def subplots(nrow=None, ncol=None, **kwargs):
    """Return an axis matrix, the user is allowed to access axes using a single number.

    If nrow and ncol are both passed in, an ordinary axis matrix is constructed, this interface allows single-index access.
    If ncol is None, we automatically calculate the grid size such that row=floor(sqrt(nrow)) and col=nrow // row (+ 1)
    """
    if nrow is None:
        fig, ax = plt.subplots()
        return fig, ax
    if ncol is None:
        row = int(np.round(np.sqrt(nrow)))
        ncol = nrow // row
        if nrow % row > 0:
            ncol += 1
        nrow = row
    fig, ax = plt.subplots(nrow, ncol, **kwargs)
    axes = np.array(ax).flatten()
    return fig, axes


def savefig(fig, fname=None, dpi=None):
    """Save a figure. If fname and dpi are None, fig is used as the name"""
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    if fname is None:
        plt.savefig(fig, dpi=dpi, bbox_inches='tight')
    else:
        fig.savefig(fname, dpi=dpi, bbox_inches='tight')


def alignXAxis(ax):
    """Given a list of axis / list of list of axis, we align the X axis."""
    if isinstance(ax[0], np.ndarray):
        axes = list(chain.from_iterable(ax))
    else:
        axes = ax
    # find min and max of xaxis
    xmin = 1e10
    xmax = -1e10
    for ax in axes:
        xmin_, xmax_ = ax.get_xlim()
        if xmin_ < xmin:
            xmin = xmin_
        if xmax_ > xmax:
            xmax = xmax_
    # ready to set them
    for ax in axes:
        ax.set_xlim(xmin, xmax)


def alignYAxis(ax):
    """Given a list of axis / list of list of axis, we align the X axis."""
    if isinstance(ax[0], np.ndarray):
        axes = list(chain.from_iterable(ax))
    else:
        axes = ax
    # find min and max of xaxis
    ymin = 1e10
    ymax = -1e10
    for ax in axes:
        ymin_, ymax_ = ax.get_xlim()
        if ymin_ < ymin:
            ymin = ymin_
        if ymax_ > ymax:
            ymax = ymax_
    # ready to set them
    for ax in axes:
        ax.set_ylim(ymin, ymax)


def alignXYAxis(ax):
    """Align both x and y axis for many axes."""
    alignXAxis(ax)
    alignYAxis(ax)


def getIndAlongAxis(x, axis, ind=0):
    if x.ndim == 1:
        return x
    slc = [slice(None)] * len(x.shape)
    slc[axis] = ind
    return x[slc]


def setFontSize(SMALL_SIZE=14):
    """change many font size for better visualization"""
    matplotlib.rc('font', size=SMALL_SIZE)
    matplotlib.rc('axes', titlesize=SMALL_SIZE)
    matplotlib.rc('axes', labelsize=SMALL_SIZE)
    matplotlib.rc('xtick', labelsize=SMALL_SIZE)
    matplotlib.rc('ytick', labelsize=SMALL_SIZE)
    matplotlib.rc('legend', fontsize=SMALL_SIZE)
    matplotlib.rc('figure', titlesize=SMALL_SIZE)
