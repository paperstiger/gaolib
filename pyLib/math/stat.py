#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Gao Tang <gt70@duke.edu>
#
# Distributed under terms of the MIT license.

"""
stat.py

Functions associated with statistics
"""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def checkstd(std, tol=1e-3):
    """Assume it is a 1 by n vector"""
    std[std < tol] = 1.0


def stdify(x, mean=None, std=None):
    """
    Given x, mean, and std, return the standarized x
    """
    if mean is None or std is None:
        mean = np.mean(x, axis=0, keepdims=True)
        std = np.std(x, axis=0, keepdims=True)
        checkstd(std)
        x = (x - mean) / std
        return x, mean, std
    else:
        return (x - mean) / std

def destdify(x, mean, std):
    """
    Recover to original x
    """
    return mean + std * x


def l1loss(x, y):
    z = np.abs(x - y)
    zz = np.zeros_like(z)
    zz[z < 1] = 0.5 * z[z < 1]**2
    zz[z >= 1] = z[z >= 1] - 0.5
    return zz


def getPCA(x, n_components=3, scale=True):
    """A quick shortcut for performing PCA.

    Parameters
    ----------
    x : ndarray, the data to be processed
    n_components : int, number of components
    scale : bool, if we apply standard scaler on x

    Returns
    -------
    x_proj : ndarray, projected data

    """
    if scale:
        x_scaled = StandardScaler().fit_transform(x)
    else:
        x_scaled = x
    pca = PCA(n_components)
    x_proj = pca.fit_transform(x_scaled)
    print(pca.explained_variance_ratio_)
    return x_proj
