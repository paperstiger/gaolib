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
import scipy.linalg
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


def _getIndAlongAxis(x, axis, ind):
    slc = [slice(None)] * x.ndim
    slc[axis] = ind
    return x[slc]


def getMeanStd(data, cols=None, axis=0):
    """Standardize a data, taken into consideration that certain data has to be scaled in the same metric.

    :param data: ndarray, the data to be standardized
    :param cols: a list or single one ndarray, the columns that have to be scaled at the same time.
    :return: mean, the mean of the dataset
    :return: std, the standard deviation of the dataset
    """
    assert axis < 2
    # check if we have to do anything
    if (isinstance(cols, bool) and not cols) or cols is None:
        return np.array([0]), np.array([1])

    # first step, we proceed as usual
    mean = np.mean(data, axis=axis, keepdims=True)
    std = np.std(data, axis=axis, keepdims=True)

    def modify_one_piece(col):
        mean_ = np.mean(_getIndAlongAxis(data, axis, col))
        std_ = np.std(_getIndAlongAxis(data, axis, col))
        if axis == 0:
            mean[0, col] = mean_
            std[0, col] = std_
        else:
            mean[col] = mean_
            std[col] = std_

    if not isinstance(cols, bool) and cols is not None:
        if isinstance(cols, np.ndarray):  # only one part
            modify_one_piece(cols)
        else:
            for col in cols:
                modify_one_piece(col)

    # finally remove those with small std
    checkstd(std)
    return mean, std


def getStandardData(data, cols=None, return_meanstd=False, axis=0):
    """Standardize the dataset, return that value.

    See getMeanStd for detailed documentation.
    """
    mean, std = getMeanStd(data, cols, axis)
    new_data = (data - mean) / std
    if not return_meanstd:
        return new_data
    else:
        return new_data, mean, std
