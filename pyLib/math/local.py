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
        :param scale: getMeanStd compatible argument
        """
        self.A_us = A
        self.B = B
        self.flann = pyflann.FLANN()
        self.querynum = qrynum
        self.checks = 16  # do not know what this means
        self.ndata = len(A)
        self.A, self.mean_, self.std_ = getStandardData(A, scale, True)
        self.params = self.flann.build_index(self.A, target_precision=0.9, log_level='info')

    def __len__(self):
        return self.ndata

    def getInd(self, x0):
        if x0.ndim == 1:
            x0 = np.expand_dims(x0, axis=0)
        x0 = stdify(x0, self.mean_, self.std_)
        if x0.dtype != self.A.dtype:
            x0 = x0.astype(self.A.dtype)
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
