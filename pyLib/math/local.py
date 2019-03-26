#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Gao Tang <gt70@duke.edu>
#
# Distributed under terms of the MIT license.

"""
local.py

Tools for analyzing local info
"""
import numpy as np
from .stat import stdify, destdify, getMeanStd, getStandardData
from sklearn.neighbors import NearestNeighbors
import numba


class Query(object):
    # A light-weighted query class.
    def __init__(self, A, B=None, qrynum=5, scale=False, distance_type='minkowski', **kw):
        """Constructor for the class.

        :param A: ndarray, features being compared against
        :param B: ndarray/None, the response
        :param qrynum: int, number of neighbors
        :param scale: getMeanStd compatible argument
        # :param distance_type: type of distances, support "euclidean" : 1, "manhattan" : 2, "minkowski" : 3, "max_dist" : 4,
        #                     "hik" : 5, "hellinger" : 6, "chi_square" : 7, "cs" : 7, "kullback_leibler" : 8, "kl" : 8,
        :param metric: see sklearn documentation for details
        :param kw: keyword arguments which include
        :algorithm:,
        :branching:
        :iterations: a number of interations
        :checks: a number of checks
        :random_seed: an seed used for initializing things
        """
        self.A_us = A
        self.B = B
        # pyflann.set_distance_type(distance_type)
        # self.flann = pyflann.FLANN(**kw)
        self.nn = NearestNeighbors(qrynum, metric=distance_type, n_jobs=-1, **kw)
        self.querynum = qrynum
        # self.checks = 16  # do not know what this means
        self.ndata = len(A)
        self.A, self.mean_, self.std_ = getStandardData(A, scale, True)
        self.nn.fit(self.A)
        # self.params = self.flann.build_index(self.A, target_precision=0.9, log_level='info')

    def __len__(self):
        return self.ndata

    def getInd(self, x0):
        if x0.ndim == 1:
            x0 = np.expand_dims(x0, axis=0)
        x0 = stdify(x0, self.mean_, self.std_)
        if x0.dtype != self.A.dtype:
            x0 = x0.astype(self.A.dtype)
        # result, dis = self.flann.nn_index(x0, self.querynum, checks=self.checks)
        dis, result = self.nn.kneighbors(x0)
        return {'index': np.squeeze(result), 'dist': np.squeeze(dis)}

    def query(self, x0):
        return self.getInd(x0)

    def getIndex(self, x0):
        return self.getInd(x0)['index']

    def getA(self, x0):
        inds = self.getInd(x0)['index']
        return self.A[inds]

    def getB(self, x0):
        if self.B is None:
            raise Exception('B is None.')
        inds = self.getInd(x0)['index']
        return self.B[inds]

    def getAB(self, x0):
        if self.B is None:
            raise Exception('B is None.')
        inds = self.getInd(x0)['index']
        return self.A[inds], self.B[inds]


def get_nn_index(x, qryx=None, n_neighbor=5, scale=True, return_dist=False, distance_type='minkowski', **kw):
    """Given a dataset, return a matrix of nearest neighbors.

    Parameters
    ----------
    x: ndarray, the dataset to be searched within.
    qryx: ndarray, the dataset that needs neighbors.
    See Query for other function arguments
    Return
    ------
    index: the index of the nearest neighbors
    dist: distance to neighbors, return only if return_dist is True
    """
    qry = Query(x, None, n_neighbor, scale, distance_type, **kw)
    if qryx is None:
        qryx = x
    rst = qry.getInd(qryx)
    if return_dist:
        return rst['index'], rst['dist']
    else:
        return rst['index']


@numba.njit(fastmath=True)
def get_affinity_matrix_xy(x, y, nn_ind, norm, rm_col_one=True, maxdx=0):
    """Construct an affinity matrix, return distances.

    :param x: ndarray, (n_sample, dim_x) the x data matrix
    :param y: ndarray, (n_sample, dim_y) the y data matrix
    :param nn_ind: ndarray, integer, (n_sample, n_neighbors) the affinity matrix
    :param rm_col_one: bool, if we remove the first column of nn_ind to save memory
    :param maxdx: float, if not 0, then distance in x larger than this value is ignored
    :return distx: ndarray, (n_sample*n_neighbors,) the distance in x
    :return disty: ndarray, (n_sample*n_neighbors,) the distance in y
    :return row: ndarray, integer, (n_sample*n_neighbors,) for constructing sparse matrix
    :return col: ndarray, integer, (n_sample*n_neighbors,) for constructing sparse matrix
    """
    n_sample = x.shape[0]
    n_neighbor = nn_ind.shape[1]
    start_col = 0
    if rm_col_one:
        n_neighbor -= 1
        start_col = 1
    if maxdx == 0:
        maxdx = np.inf
    len_dist = n_sample * n_neighbor
    distx = np.zeros(len_dist)
    disty = np.zeros(len_dist)
    out_row = np.zeros(len_dist, dtype=np.int64)
    out_col = np.zeros(len_dist, dtype=np.int64)
    # loop over the data
    idx = 0
    for i in range(n_sample):
        for j in range(start_col, n_neighbor + start_col):
            dist_in_x = np.linalg.norm(x[i] - x[nn_ind[i, j]], norm)
            if dist_in_x > maxdx:
                continue
            out_row[idx] = i
            out_col[idx] = nn_ind[i, j]
            distx[idx] = dist_in_x
            disty[idx] = np.linalg.norm(y[i] - y[out_col[idx]], norm)
            idx += 1
    return distx[:idx], disty[:idx], out_row[:idx], out_col[:idx]
