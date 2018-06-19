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
from sklearn.preprocessing import StandardScaler
try:
    import pyflann
except:
    pass


class Query(object):
    # A light-weighted query class.
    def __init__(self, A, B=None, qrynum=5, scale=False):
        """Constructor for the class.

        :param A: ndarray, features being compared against
        :param B: ndarray/None, the response
        :param qrynum: int, number of neighbors
        :param scale: bool, if we use standard scaler
        """
        self.A = A
        self.B = B
        self.flann = pyflann.FLANN()
        self.params = self.flann.build_index(A, target_precision=0.9, log_level='info')
        self.querynum = qrynum
        self.checks = 16  # do not know what this means
        self.ndata = len(self.A)
        self.scale = scale
        if scale:
            self.scaler = StandardScaler()
            self.A = self.scaler.fit_transform(A)

    def __len__(self):
        return self.ndata

    def getInd(self, x0):
        if x0.ndim == 1:
            x0 = np.expand_dims(x0, axis=0)
        if x0.dtype != self.A.dtype:
            x0 = x0.astype(self.A.dtype)
        if self.scale:
            x0 = self.scaler.transform(x0)
        result, dis = self.flann.nn_index(x0, self.querynum, checks=self.checks)
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
