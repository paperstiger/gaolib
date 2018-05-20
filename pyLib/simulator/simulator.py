#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Gao Tang <gt70@duke.edu>
#
# Distributed under terms of the MIT license.

"""
simulator.py
A simulator
"""
from scipy.integrate import ode
import numpy as np


class TrajPropagate(object):
    def __init__(self, sys, method='c', *args, **kw):
        self.sys = sys
        self.dimx, self.dimu = sys.dimx, sys.dimu
        self.int_method = method
        if self.int_method == 'c':
            self.ode45 = ode(self.dynwrapper).set_integrator('dopri5')
        else:
            self.ode45 = None

    def dynwrapper(self, t, y, u):
        return self.sys.getDf(t, y, u)

    def onestep(self, x0, ctrl, t):
        """Assume we use constant control for one step, see outcome."""
        assert len(t) == 2
        if self.int_method == 'c':
            self.ode45.set_initial_value(x0, t[0]).set_f_params(ctrl)
            return self.ode45.integrate(t[1])
        else:
            dx = self.dynwrapper(t[0], x0, ctrl)
            return x0 + dx * (t[1] - t[0])


class Simulator(TrajPropagate):
    """A simulator that supports controller"""
    def __init__(self, sys, method='c', ctrl=None, dt=0.02, *args, **kw):
        super(Simulator, self).__init__(sys, method, *args, **kw)
        self.ctrller = ctrl
        self.dt = dt
        self.x0 = np.zeros(self.dimx)
        self.t0 = 0.
        self.statefun = None
        self.x0fun = None
        self.observer = None
        self.perturber = None

    def setX0(self, x0):
        self.x0 = x0

    def simulate(self, tspan):
        self.t0 = tspan[0]
        Nstep = int(np.ceil((tspan[1] - tspan[0])/self.dt))
        curt = self.t0
        state = self.x0
        status = 0
        vX, vU = [], []
        vdX = []
        for stepi in range(Nstep):
            curt = self.t0 + stepi * self.dt
            if self.observer is not None:
                ctrl = self.ctrller(curt, self.observer(state))
            else:
                ctrl = self.ctrller(curt, state)
            vX.append(state)
            vU.append(ctrl)
            if self.statefun is not None:
                status = self.statefun(curt, state)
                if status != 0:
                    break
            state = self.onestep(state, ctrl, [curt, curt + self.dt])
            if self.perturber is not None:
                state = self.perturber(state)
        vX, vU = np.array(vX), np.array(vU)
        vt = self.t0 + self.dt * np.arange(stepi + 1)
        return {'status': status, 'statef': state, 'vt': vt, 'vX': vX, 'vU': vU}
