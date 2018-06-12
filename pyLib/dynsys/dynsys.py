#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Gao Tang <gt70@duke.edu>
#
# Distributed under terms of the MIT license.

"""
dynsys.py
"""
import numpy as np
from math import sin, cos, sqrt
# import numba


class dynsys(object):
    """Base class for dynamical system"""
    def __init__(self, dimx, dimu):
        self.dimx = dimx
        self.dimu = dimu

    def getDf(self, t, x, u):
        """Given t, x, u, return dxdt"""
        raise NotImplementedError

    def getJac(self, t, x, u):
        """Given t, x, u, return Jx, Ju"""
        raise NotImplementedError

    def getAB(self, t, x, u, dt):
        """Given t, x, u, return A, B"""
        Jx, Ju = self.getJac(t, x, u)
        Jx = Jx * dt + np.eye(self.dimx)
        Ju = Ju * dt
        return Jx, Ju


class Pendulum(dynsys):
    """Demo class, the pendulum. We assume gravity is 1"""
    def __init__(self):
        super(Pendulum, self).__init__(2, 1)

    def getDf(self, t, x, u):
        sta = np.sin(x[0])
        f = np.zeros(self.dimx)
        f[0] = x[1]
        f[1] = u[0] - sta
        return f

    def getJac(self, t, x, u):
        """Return Jx and Ju, cts case since dt is unknown"""
        Jx = np.zeros((self.dimx, self.dimx))
        Ju = np.zeros((self.dimx, self.dimu))
        cta = np.cos(x[0])
        Jx[0, 1] = 1.0
        Jx[1, 0] = -cta
        Ju[1, 0] = 1
        return Jx, Ju

    def getfg(self, x):
        f = np.zeros(self.dimx)
        g = np.zeros((self.dimx, self.dimu))
        f[0] = x[1]
        f[1] = -np.sin(x[0])
        g[1, 0] = 1.0
        return f, g


class SecondOrderPlanarCar(dynsys):
    """A second order planar car."""
    def __init__(self):
        dynsys.__init__(self, 4, 2)

    def getDf(self, t, x, u):
        return np.concatenate((x[2:], u))

    def getJac(self, t, x, u):
        Jx = np.zeros((self.dimx, self.dimx))
        Ju = np.zeros((self.dimx, self.dimu))
        Jx[0, 2] = 1.0
        Jx[1, 3] = 1.0
        Ju[2, 0] = 1.0
        Ju[3, 1] = 1.0
        return Jx, Ju


class DubinCar(dynsys):
    """Demo class, the pendulum"""
    def __init__(self):
        dynsys.__init__(self, 4, 2)

    def getDf(self, t, x, u):
        sta = sin(x[2])
        cta = cos(x[2])
        v = x[3]
        f = np.zeros(self.dimx)
        f[0] = v * sta
        f[1] = v * cta
        f[2] = u[0] * v
        f[3] = u[1]
        return f

    def getfg(obj, x):
        sta = sin(x[2])
        cta = cos(x[2])
        v = x[3]
        f = np.zeros(4)
        g = np.zeros((4, 2))
        f[0] = v * sta
        f[1] = v * cta
        f[2] = 0
        f[3] = 0
        g[2, 0] = v
        g[3, 1] = 1
        return f, g

    def getJac(obj, t, x, u):
        """Return Jx and Ju, cts case since dt is unknown"""
        Jx = np.zeros((obj.dimx, obj.dimx))
        Ju = np.zeros((obj.dimx, obj.dimu))
        sta, cta, v = sin(x[2]), cos(x[2]), x[3]
        Jx[0, 3] = sta
        Jx[0, 2] = v * cta
        Jx[1, 3] = cta
        Jx[1, 2] = -v * sta
        Jx[2, 3] = u[0]
        Ju[2, 0] = v
        Ju[3, 1] = 1
        return Jx, Ju


class QuadCopter(dynsys):
    """Demo class, for a quadcopter"""
    def __init__(self, quat=False):
        super(QuadCopter, self).__init__(12, 4)
        self.m = 0.5
        self.g = 9.81
        self.kF = 1.
        self.kM = 0.0245
        self.L = 0.175
        self.In = np.array([0.0023, 0.0023, 0.004])
        self.quat = quat

    def getDf(self, t, x, u):
        """Return time-derivative of state"""
        if self.quat:
            return quatQuadDyn(x, u, self.m, self.g, self.kF, self.kM, self.L, self.In)
        else:
            return quadDyn(x, u, self.m, self.g, self.kF, self.kM, self.L, self.In)

    def getJac(self, t, x, u):
        """Return Jx and Ju"""
        if self.quat:
            return quatQuadJac(x, u, self.m, self.g, self.kF, self.kM, self.L, self.In)
        else:
            return quadJac(x, u, self.m, self.g, self.kF, self.kM, self.L, self.In)

    def getfg(self, x):  # a weird interface
        return quadFG(x, self.m, self.g, self.kF, self.kM, self.L, self.In)


#@numba.jit
def quadDyn(x, u, m, g, kF, kM, L, In):
    """Dyn fun for quadcopter"""
    phi, theta, psi = x[3], x[4], x[5]
    xd, yd, zd = x[6], x[7], x[8]
    p, q, r = x[9], x[10], x[11]
    t1 = cos(theta)
    t2 = sin(theta)
    t3 = sin(phi)
    t4 = cos(phi)
    t5 = 0.1e1 / t4
    t5 = t5 * (p * t2 - r * t1)
    t6 = cos(psi)
    t7 = sin(psi)
    t8 = t1 * t3
    t9 = (u[0] + u[1] + u[2] + u[3]) * kF
    t10 = 0.1e1 / m
    f = np.array([xd,yd,zd,p * t1 + r * t2,t5 * t3 + q,-t5,t10 * (t2 * t6 + t8 * t7) * t9,t10 * (t2 * t7 - t8 * t6) * t9,t10 * t4 * t1 * t9 - g,-0.1e1 / In[0] * (-L * kF * (u[1] - u[3]) + q * r * (-In[1] + In[2])),-0.1e1 / In[1] * (L * kF * (u[0] - u[2]) + p * r * (In[0] - In[2])),0.1e1 / In[2] * (p * q * (In[0] - In[1]) + (u[0] - u[1] + u[2] - u[3]) * kM)])
    return f


def quatQuadDyn(x, u, m, g, kF, kM, L, In):
    """Dyn equation for quadcopter in quaternion"""
    qx, qy, qz = x[3], x[4], x[5]
    xd, yd, zd = x[6], x[7], x[8]
    p, q, r = x[9], x[10], x[11]
    t1 = qx ** 2
    t2 = qy ** 2
    t3 = -qz ** 2 - t1 - t2 + 1
    t3 = np.sqrt(t3)
    t4 = 0.1e1 / 0.2e1
    t5 = (u[0] + u[1] + u[2] + u[3]) * kF
    t6 = 0.1e1 / m
    t7 = 2 * t6
    f = np.array([xd,yd,zd,t4 * (p * qy + q * qz - r * t3),-t4 * (p * qx - q * t3 - r * qz),-t4 * (p * t3 + q * qx + r * qy),t7 * (qx * qz + qy * t3) * t5,t7 * (-qx * t3 + qy * qz) * t5,-g + t6 * (-2 * t2 - 2 * t1 + 1) * t5,-1 / In[0] * (-L * kF * (u[1] - u[3]) + q * r * (-In[1] + In[2])),1 / In[1] * (-L * kF * (u[0] - u[2]) + p * r * (-In[0] + In[2])),1 / In[2] * (p * q * (In[0] - In[1]) + (u[0] - u[1] + u[2] - u[3]) * kM)])
    return f


def quatQuadJac(x, u, m, g, kF, kM, L, In):
    """Return Jx and Ju, in quaternion case"""
    qx, qy, qz = x[3], x[4], x[5]
    xd, yd, zd = x[6], x[7], x[8]
    p, q, r = x[9], x[10], x[11]
    t1 = qx ** 2
    t2 = qy ** 2
    t3 = -qz ** 2 - t1 - t2 + 1
    t4 = t3 ** (-0.1e1 / 0.2e1)
    t5 = r * t4
    t3 = t3 * t4
    t6 = 0.1e1 / 0.2e1
    t7 = t6 * qy
    t8 = t6 * qz
    t9 = t6 * t3
    t10 = t6 * qx
    t11 = q * t4
    t12 = p * t4
    t13 = t4 * qy
    t14 = t13 * qx
    t15 = (u[0] + u[1] + u[2] + u[3]) * kF
    t16 = qx * qz
    t17 = 0.1e1 / m
    t18 = 2 * t17
    t19 = t18 * (qy * t3 + t16) * kF
    t20 = t18 * (-qx * t3 + qy * qz) * kF
    t21 = t17 * (-2 * t1 - 2 * t2 + 1) * kF
    t17 = -4 * t17
    t22 = In[1] - In[2]
    t23 = 0.1e1 / In[0]
    t24 = t23 * L * kF
    t25 = -In[0] + In[2]
    t26 = 0.1e1 / In[1]
    t27 = t26 * L * kF
    t28 = In[0] - In[1]
    t29 = 0.1e1 / In[2]
    t30 = t29 * kM
    cg = np.array([[0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],[0,0,0,t10 * t5,t6 * (t5 * qy + p),t6 * (t5 * qz + q),0,0,0,t7,t8,-t9,0,0,0,0],[0,0,0,-t6 * (t11 * qx + p),-t7 * t11,t6 * (-t11 * qz + r),0,0,0,-t10,t9,t8,0,0,0,0],[0,0,0,-t6 * (-t12 * qx + q),-t6 * (-t12 * qy + r),t8 * t12,0,0,0,-t9,-t10,-t7,0,0,0,0],[0,0,0,t18 * (-t14 + qz) * t15,t18 * (-t2 * t4 + t3) * t15,t18 * (-t13 * qz + qx) * t15,0,0,0,0,0,0,t19,t19,t19,t19],[0,0,0,-t18 * (-t1 * t4 + t3) * t15,t18 * (t14 + qz) * t15,t18 * (t16 * t4 + qy) * t15,0,0,0,0,0,0,t20,t20,t20,t20],[0,0,0,t17 * qx * t15,t17 * qy * t15,0,0,0,0,0,0,0,t21,t21,t21,t21],[0,0,0,0,0,0,0,0,0,0,t23 * r * t22,t23 * q * t22,0,t24,0,-t24],[0,0,0,0,0,0,0,0,0,t26 * r * t25,0,t26 * p * t25,-t27,0,t27,0],[0,0,0,0,0,0,0,0,0,t29 * q * t28,t29 * p * t28,0,t30,-t30,t30,-t30]])
    Jx = cg[:, :12]
    Ju = cg[:, 12:]
    return Jx, Ju


#@numba.jit
def quadJac(x, u, m, g, kF, kM, L, In):
    """Return Jx and Ju, cts case since dt is unknown"""
    phi, theta, psi = x[3], x[4], x[5]
    xd, yd, zd = x[6], x[7], x[8]
    p, q, r = x[9], x[10], x[11]
    t1 = sin(theta)
    t2 = cos(theta)
    t3 = t2 * r
    t4 = t1 * p
    t5 = -t3 + t4
    t6 = sin(phi)
    t7 = cos(phi)
    t8 = 0.1e1 / t7
    t9 = t8 ** 2
    t10 = t6 ** 2 * t9 + 1
    t11 = t8 * (p * t2 + r * t1)
    t12 = t2 * t8
    t8 = t1 * t8
    t13 = sin(psi)
    t14 = (u[0] + u[1] + u[2] + u[3]) * kF
    t15 = cos(psi)
    t16 = t13 * t1
    t17 = t15 * t2
    t18 = -t17 * t6 + t16
    t13 = t2 * t13
    t15 = t15 * t1
    t19 = 0.1e1 / m
    t20 = t19 * (t13 * t6 + t15)
    t21 = t20 * kF
    t22 = t19 * t18 * kF
    t23 = t19 * t7
    t24 = t23 * t2 * kF
    t25 = In[1] - In[2]
    t26 = 0.1e1 / In[0]
    t27 = t26 * L * kF
    t28 = -In[0] + In[2]
    t29 = 0.1e1 / In[1]
    t30 = t29 * L * kF
    t31 = In[0] - In[1]
    t32 = 0.1e1 / In[2]
    t33 = t32 * kM
    cg0 = np.array([[0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],[0,0,0,0,-t5,0,0,0,0,t2,0,t1,0,0,0,0],[0,0,0,-t3 * t10 + t4 * t10,t11 * t6,0,0,0,0,t8 * t6,1,-t12 * t6,0,0,0,0],[0,0,0,-t9 * t6 * t5,-t11,0,0,0,0,-t8,0,t12,0,0,0,0],[0,0,0,t13 * t19 * t7 * t14,t19 * (-t16 * t6 + t17) * t14,-t19 * t18 * t14,0,0,0,0,0,0,t21,t21,t21,t21],[0,0,0,-t17 * t19 * t7 * t14,t19 * (t15 * t6 + t13) * t14,t20 * t14,0,0,0,0,0,0,t22,t22,t22,t22],[0,0,0,-t19 * t6 * t2 * t14,-t23 * t1 * t14,0,0,0,0,0,0,0,t24,t24,t24,t24],[0,0,0,0,0,0,0,0,0,0,t26 * r * t25,t26 * q * t25,0,t27,0,-t27],[0,0,0,0,0,0,0,0,0,t29 * r * t28,0,t29 * p * t28,-t30,0,t30,0],[0,0,0,0,0,0,0,0,0,t32 * q * t31,t32 * p * t31,0,t33,-t33,t33,-t33]])
    Jx, Ju = cg0[:, :12], cg0[:, 12:]
    return Jx, Ju


#@numba.jit
def quadFG(x, m, g, kF, kM, L, In):
    phi, theta, psi = x[3], x[4], x[5]
    xd, yd, zd = x[6], x[7], x[8]
    p, q, r = x[9], x[10], x[11]
    t1 = cos(theta)
    t2 = sin(theta)
    t3 = sin(phi)
    t4 = cos(phi)
    t5 = p * t2 - r * t1
    t6 = 0.1e1 / t4
    t7 = cos(psi)
    t8 = sin(psi)
    t9 = t1 * t3
    t10 = 0.1e1 / m
    t11 = t10 * (t2 * t7 + t9 * t8) * kF
    t7 = t10 * (t2 * t8 - t9 * t7) * kF
    t8 = t10 * t4 * t1 * kF
    t9 = 0.1e1 / In[0]
    t10 = t9 * L * kF
    t12 = 0.1e1 / In[1]
    t13 = t12 * L * kF
    t14 = 0.1e1 / In[2]
    t15 = t14 * kM
    cg2 = np.array([[xd,0,0,0,0],[yd,0,0,0,0],[zd,0,0,0,0],[p * t1 + r * t2,0,0,0,0],[(q * t4 + t3 * t5) * t6,0,0,0,0],[-t5 * t6,0,0,0,0],[0,t11,t11,t11,t11],[0,t7,t7,t7,t7],[-g,t8,t8,t8,t8],[q * r * (In[1] - In[2]) * t9,0,t10,0,-t10],[p * r * (In[2] - In[0]) * t12,-t13,0,t13,0],[-p * q * (In[1] - In[0]) * t14,t15,-t15,t15,-t15]])
    f = cg2[:, 0]
    g = cg2[:, 1:]
    return f, g
