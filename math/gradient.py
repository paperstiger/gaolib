#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Gao Tang <gt70@duke.edu>
#
# Distributed under terms of the MIT license.

"""
gradient.py

Tools for calculating gradients based solely on data.
Monte-Carlo like approach and only return the norm.
"""
import numpy as np
import pyflann


def gradNorm(x, y, knn=5, dsrate=1):
    """
    Given dataset x-y, try to estimate the gradients.
    knn is # neighbors for gradient estimation
    dsrate is downsampling rate, used in eval not build
    """
    assert x.ndim <=2 and y.ndim <= 2
    if x.ndim == 1:
        x = np.expand_dims(x, axis=1)
    if y.ndim == 1:
        y = np.expand_dims(y, axis=1)
    nX = x.shape[0]
    assert nX == y.shape[0]
    flann = pyflann.FLANN()
    params = flann.build_index(x)
    selfchecks = 16
    sz = int(dsrate*nX)
    grad = np.zeros(sz)
    # loop over sampled x
    if dsrate == 1:
        inds = np.arange(nX)
    else:
        inds = np.random.choice(nX, size=int(dsrate*nX), replace=False)  # so unique
    # get all neighbors
    index, dis = flann.nn_index(x[inds], knn)
    for i, ind in enumerate(inds):
        vdx = x[index[i, 1:]] - x[index[i, 0]]
        dx = np.linalg.norm(vdx, axis=1)
        vdy = y[index[i, 1:]] - y[index[i, 0]]
        dy = np.linalg.norm(vdy, axis=1)
        grad[i] = np.mean(dy / dx)  # assume no dx equals 0
    return inds, grad


def verboseGradNorm(x, y, knn=5, dsrate=1):
    """Get complete information for gradient of the field.
    Aiming at detecting discontinuity.
    """
    assert x.ndim <=2 and y.ndim <= 2
    if x.ndim == 1:
        x = np.expand_dims(x, axis=1)
    if y.ndim == 1:
        y = np.expand_dims(y, axis=1)
    nX = x.shape[0]
    assert nX == y.shape[0]
    flann = pyflann.FLANN()
    params = flann.build_index(x)
    # what I do is for each lucky one, estimate several gradients
    sz = int(dsrate*nX)
    grad = np.zeros((sz, knn - 1))
    dx = np.zeros((sz, knn - 1))
    dy = np.zeros((sz, knn - 1))
    # loop over sampled x
    if dsrate == 1:
        inds = np.arange(nX)
    else:
        inds = np.random.choice(nX, size=int(dsrate*nX), replace=False)  # so unique
    # get all neighbors
    index, dis = flann.nn_index(x[inds], knn)
    for i, ind in enumerate(inds):
        vdx = x[index[i, 1:]] - x[index[i, 0]]
        dx_ = np.linalg.norm(vdx, axis=1)
        vdy = y[index[i, 1:]] - y[index[i, 0]]
        dy_ = np.linalg.norm(vdy, axis=1)
        grad_ = dy_ / dx_  # assume no dx equals 0
        dx[i] = dx_
        dy[i] = dy_
        grad[i] = grad_
    return {'index': inds, 'nn': index, 'dx': dx, 'dy': dy, 'grad': grad}
