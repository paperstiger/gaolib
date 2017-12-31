#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Gao Tang <gt70@duke.edu>
#
# Distributed under terms of the MIT license.

"""
lqr.py

A class to simulate LQR controller.
"""
import numpy as np
from controller import linearFeedbackController, feedForwardBackward, openLoopController
from control import care, dare
from trajectory import zeroOrderHolder


def tvlqr(A, B, Q, R, F):
    """solve tvlqr problem iteratively"""
    n = len(A)
    (dimx, dimu) = B[0].shape
    vK = [np.zeros((dimu, dimx)) for i in range(n)]
    vP = [np.zeros((dimx, dimx)) for i in range(n + 1)]
    vP[n] = F
    for i in xrange(n):
        Ai = A[n - 1 - i]
        Bi = B[n - 1 - i]
        Pip1 = vP[n - i]
        TBPB = R + Bi.T.dot(Pip1).dot(Bi)
        rhs = (Bi.T).dot(Pip1.dot(Ai))
        K = np.linalg.solve(TBPB, rhs)
        vK[n - 1 - i] = K
        C = Ai - Bi.dot(K)
        vP[n - 1 - i] = Q + (K.T).dot(R).dot(K) + (C.T).dot(Pip1).dot(C)
    return vK, vP


class ABQRLQR(linearFeedbackController):
    """An LQR with given A, B, Q, R. We use continuous case"""
    def __init__(self, A, B, Q, R, method='c', x0=None, u0=None, ulb=None, uub=None):
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        if method == 'c':
            X, L, G = care(A, B, Q, R)
        else:
            X, L, G = dare(A, B, Q, R)
        if isinstance(G, np.matrix):
            G = np.array(G)
        # check eigenvalues of A-BK
        print 'Eigen values of care are {}'.format(L)
        super(ABQRLQR, self).__init__(G, x0, u0, ulb, uub)


class stabilizeLQR(ABQRLQR):
    """A LQR controller which stabilize the system around a fixed point"""
    def __init__(self, sys, Q, R, x0, u0, method='c', dt=None, ulb=None, uub=None):
        if method == 'c':
            A, B = sys.getJac(0, x0, u0)
        else:
            A, B = sys.getAB(0, x0, u0, dt)
        super(stabilizeLQR, self).__init__(A, B, Q, R, method, x0, u0, ulb, uub)


class trackLQR(feedForwardBackward):
    """An LQR controller to track a trajectory"""
    def __init__(self, sys, traj, Q, R, F, ulb=None, uub=None):
        # get scheduled feedback LQR
        N, dimx, dimu = traj.N, traj.dimx, traj.dimu
        vA = np.zeros((N-1, dimx, dimx))
        vB = np.zeros((N-1, dimx, dimu))
        dt = traj.dt
        for i in xrange(N - 1):
            vA[i], vB[i] = sys.getAB(0, traj.X[i], traj.U[i], dt)
        vK, vP = tvlqr(vA, vB, Q, R, F)
        forward = openLoopController(traj, ulb=None, uub=None)
        backward = linearFeedbackController(np.zeros((dimu, dimx)), x0=None, u0=None, ulb=None, uub=None)  # K is unset yet
        super(trackLQR, self).__init__(forward, backward, ulb, uub)
        self.dt = traj.dt
        self.scheduleK = zeroOrderHolder(vK, self.dt)

    def __call__(self, t, x):
        self.backward.K = self.scheduleK.getObj(t)
        u = super(trackLQR, self).__call__(t, x)
        u = self.trim(u)
        return u
