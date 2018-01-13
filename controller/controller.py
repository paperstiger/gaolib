#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Gao Tang <gt70@duke.edu>
#
# Distributed under terms of the MIT license.

"""
controller.py
Basic class necessary for a controller
"""
import numpy as np
import logging


logger = logging.getLogger(__name__)


class controller(object):
    """Base class for any controller"""
    def __init__(self, dimx, dimu, ulb=None, uub=None):
        self.dimx = dimx
        self.dimu = dimu
        self.ulb = ulb
        self.uub = uub

    def setUbd(self, ulb, uub):
        assert ulb is not None and uub is not None
        self.ulb, self.uub = ulb, uub

    def trim(self, u):
        if self.ulb is not None:
            u = np.maximum(self.ulb, np.minimum(self.uub, u))
        return u

    def __call__(self, *args, **kw):
        """Given x return u"""
        raise NotImplementedError


class linearFeedbackController(controller):
    """Controller with linear feedback."""
    def __init__(self, K, x0=None, u0=None, ulb=None, uub=None):
        dimu, dimx = K.shape
        super(linearFeedbackController, self).__init__(dimx, dimu, ulb, uub)
        self.K = K
        if x0 is None:
            self.x0 = np.zeros(shape=self.dimx)
        else:
            self.x0 = x0
        if u0 is None:
            self.u0 = np.zeros(shape=self.dimu)
        else:
            self.u0 = u0

    def __call__(self, t, x):
        u = self.u0 - self.K.dot(x - self.x0)
        return self.trim(u)


class nonlinearFunController(controller):
    """A controller that basically evaluate a function."""
    def __init__(self, fun, dimx, dimu, ulb=None, uub=None):
        super(nonlinearFunController, self).__init__(dimx, dimu, ulb, uub)
        self.fun = fun

    def __call__(self, t, x):
        u = self.fun(t, x)
        return self.trim(u)


class openLoopController(controller):
    """Open loop controller, it can be incorporated with feedback controller"""
    def __init__(self, traj, ulb=None, uub=None):
        self.traj = traj
        dimx, dimu = traj.dimx, traj.dimu
        super(openLoopController, self).__init__(dimx, dimu, ulb, uub)

    def getxu(self, t):
        x, u = self.traj(t)
        u = self.trim(u)
        return x, u

    def __call__(self, t):
        x, u = self.getxu(t)
        return u


class feedForwardBackward(controller):
    """A controller with both feedforward and feedback terms"""
    def __init__(self, forward, backward, ulb=None, uub=None):
        assert forward.dimx == backward.dimx and forward.dimu == backward.dimu
        super(feedForwardBackward, self).__init__(forward.dimx, forward.dimu, ulb, uub)
        assert forward.ulb is None and backward.ulb is None
        self.forward = forward
        self.backward = backward

    def __call__(self, t, x):
        xref, uref = self.forward.getxu(t)
        uback = self.backward(t, x - xref)
        u = uref + uback
        logger.debug('xref {}\n dx {}\n uref {} uback {} u {}'.format(xref, x - xref, uref, uback, u))
        return self.trim(u)
