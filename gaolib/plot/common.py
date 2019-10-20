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
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import subprocess
import tempfile
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
    fig.tight_layout()
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
    return x[tuple(slc)]


def setFontSize(SMALL_SIZE=14):
    """change many font size for better visualization"""
    matplotlib.rc('font', size=SMALL_SIZE)
    matplotlib.rc('axes', titlesize=SMALL_SIZE)
    matplotlib.rc('axes', labelsize=SMALL_SIZE)
    matplotlib.rc('xtick', labelsize=SMALL_SIZE)
    matplotlib.rc('ytick', labelsize=SMALL_SIZE)
    matplotlib.rc('legend', fontsize=SMALL_SIZE)
    matplotlib.rc('figure', titlesize=SMALL_SIZE)


"""The following functions are stolen from elsewhere"""

def get_fig_size(fig_width_cm, fig_height_cm=None):
    """Convert dimensions in centimeters to inches.
    If no height is given, it is computed using the golden ratio.
    """
    if not fig_height_cm:
        golden_ratio = (1 + math.sqrt(5))/2
        fig_height_cm = fig_width_cm / golden_ratio

    size_cm = (fig_width_cm, fig_height_cm)
    return map(lambda x: x/2.54, size_cm)


"""
The following functions can be used by scripts to get the sizes of
the various elements of the figures.
"""


def label_size():
    """Size of axis labels
    """
    return 10


def font_size():
    """Size of all texts shown in plots
    """
    return 10


def ticks_size():
    """Size of axes' ticks
    """
    return 8


def axis_lw():
    """Line width of the axes
    """
    return 0.6


def plot_lw():
    """Line width of the plotted curves
    """
    return 1.5


def fig_setup():
    """Set all the sizes to the correct values and use
    tex fonts for all texts.
    """
    params = {'text.usetex': True,
              'figure.dpi': 200,
              'font.size': font_size(),
              'font.serif': [],
              'font.sans-serif': [],
              'font.monospace': [],
              'axes.labelsize': label_size(),
              'axes.titlesize': font_size(),
              'axes.linewidth': axis_lw(),
              'text.fontsize': font_size(),
              'legend.fontsize': font_size(),
              'xtick.labelsize': ticks_size(),
              'ytick.labelsize': ticks_size(),
              'font.family': 'serif'}
    plt.rcParams.update(params)


def save_fig(fig, file_name, fmt=None, dpi=300, tight=True):
    """Save a Matplotlib figure as EPS/PNG/PDF to the given path and trim it.
    """

    if not fmt:
        fmt = file_name.strip().split('.')[-1]

    if fmt not in ['eps', 'png', 'pdf']:
        raise ValueError('unsupported format: %s' % (fmt,))

    extension = '.%s' % (fmt,)
    if not file_name.endswith(extension):
        file_name += extension

    file_name = os.path.abspath(file_name)
    with tempfile.NamedTemporaryFile() as tmp_file:
        tmp_name = tmp_file.name + extension

    # save figure
    if tight:
        fig.savefig(tmp_name, dpi=dpi, bbox_inches='tight')
    else:
        fig.savefig(tmp_name, dpi=dpi)

    # trim it
    if fmt == 'eps':
        subprocess.call('epstool --bbox --copy %s %s' %
                        (tmp_name, file_name), shell=True)
    elif fmt == 'png':
        subprocess.call('convert %s -trim %s' %
                        (tmp_name, file_name), shell=True)
    elif fmt == 'pdf':
        subprocess.call('pdfcrop %s %s' % (tmp_name, file_name), shell=True)
