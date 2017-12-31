#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Gao Tang <gt70@duke.edu>
#
# Distributed under terms of the MIT license.

"""
trajectory.py
Mathematically define a trajectory.
"""
import numpy as np
from scipy.interpolate import interp1d
from numbers import Number


class trajectory(object):
    """A class for trajectory."""
    def __init__(self, dimx, dimu):
        self.dimx = dimx
        self.dimu = dimu

    def __call__(self, t):
        raise NotImplementedError


class zeroOrderHolder(object):
    """Looks like trajectory, but a time scheduled things."""
    def __init__(self, obj, dt, t0=0, *args, **kw):
        self.obj = obj
        self.N = len(self.obj)
        self.dt = dt
        self.t0 = t0
        if 'objf' in kw:
            self.objf = kw['objf']
        else:
            self.objf = None

    def setObjf(self, objf):
        self.objf = objf

    def getObj(self, t):
        """Get the obj at t"""
        if t < self.t0 or t > self.N*self.dt:
            if t < self.t0:
                return self.obj[0]
            else:
                if self.objf is None:
                    return self.obj[-1]
                else:
                    return self.objf
        else:
            ind = int((t - self.t0) // self.dt)
            if ind == self.N:
                ind = self.N - 1
            return self.obj[ind]


class interpTrajectory(trajectory):
    """A class for trajectory with interpolation"""
    def __init__(self, mx, mu, vt, *args, **kw):
        assert mx.ndim == 2 and mu.ndim == 2
        Nx, dimx = mx.shape
        Nu, dimu = mu.shape
        assert Nx == Nu or Nx == Nu + 1
        self.N = Nx
        super(interpTrajectory, self).__init__(dimx, dimu)
        if isinstance(vt, Number):
            self.vt = np.linspace(0, vt, self.N)
        if isinstance(vt, (list, tuple)):
            self.vt = np.linspace(vt[0], vt[1], self.N)
        if isinstance(vt, np.ndarray):
            assert vt.ndim == 1
            if vt.size == 2:
                self.vt = np.linspace(vt[0], vt[1], self.N)
            elif vt.size == Nx:
                self.vt = vt
            else:
                raise Exception("Incorrect vt")
        self.X = mx
        if Nx == Nu:
            self.U = mu
        else:
            self.U = np.concatenate((mu, mu[-1:]), axis=0)
        # create interpolation instance
        self.interpX = [interp1d(self.vt, self.X[:, i], assume_sorted=True, copy=False, *args, **kw) for i in xrange(self.dimx)]
        self.interpU = [interp1d(self.vt, self.U[:, i], assume_sorted=True, copy=False, *args, **kw) for i in xrange(self.dimu)]
        self.tfFun = None
        self.exceedTf = False

    def __call__(self, t):
        if t >= self.vt[0] and t <= self.vt[-1]:
            x = np.array([intx(t) for intx in self.interpX])
            u = np.array([intu(t) for intu in self.interpU])
            return x, u
        else:
            if not self.exceedTf:
                print 'Warning: time {} is not within bound [{}, {}] at {}'.format(t, self.vt[0], self.vt[-1], __file__)
                self.exceedTf = True
            if t >= self.vt[-1]:
                if self.tfFun is None:
                    return self.X[-1], self.U[-1]
                else:
                    return self.tfFun(t)
            else:
                return self.X[0], self.U[0]

    def evaltf(self, t):
        """Return a value after tf is reached"""
        if self.tfFun is not None:
            return self.tfFun(t)

    @property
    def dt(self):
        return self.vt[1] - self.vt[0]
