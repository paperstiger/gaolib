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
from math import sin, cos
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


class pendulum(dynsys):
    """Demo class, the pendulum"""
    def __init__(self):
        super(pendulum, self).__init__(2, 1)

    def getDf(self, t, x, u):
        sta = np.sin(x[0])
        f = np.zeros(self.dimx)
        f[0] = x[1]
        f[1] = u[0] - sta
        return f

    def getJac(obj, t, x, u):
        """Return Jx and Ju, cts case since dt is unknown"""
        Jx = np.zeros((obj.dimx, obj.dimx))
        Ju = np.zeros((obj.dimx, obj.dimu))
        cta = np.cos(x[0])
        Jx[0, 1] = 1.0
        Jx[1, 0] = -cta
        Ju[1, 0] = 1
        return Jx, Ju

    def getfg(obj, x):
        f = np.zeros(obj.dimx)
        g = np.zeros((obj.dimx, obj.dimu))
        f[0] = x[1]
        f[1] = -np.sin(x[0])
        g[1, 0] = 1.0
        return f, g


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


class quadCopter(dynsys):
    """Demo class, for a quadcopter"""
    def __init__(self, quat=False):
        super(quadCopter, self).__init__(12, 4)
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
    qw, qx, qy, qz = x[3], x[4], x[5], x[6]
    xd, yd, zd = x[7], x[8], x[9]
    p, q, r = x[10], x[11], x[12]
    t1 = 0.1e1 / 0.2e1
    t2 = (u[0] + u[1] + u[2] + u[3]) * kF
    t3 = 0.1e1 / m
    t4 = 2 * t3
    f = np.array([xd,yd,zd,t1 * (p * qz - q * qy + r * qx),-t1 * (-p * qy - q * qz + r * qw),t1 * (-p * qx + q * qw + r * qz),-t1 * (p * qw + q * qx + r * qy),t4 * (qw * qy + qx * qz) * t2,-t4 * (qw * qx - qy * qz) * t2,t3 * (-2 * qx ** 2 - 2 * qy ** 2 + 1) * t2 - g,0.1e1 / In[0] * (L * kF * (u[1] - u[3]) + q * r * (In[1] - In[2])),-0.1e1 / In[1] * (L * kF * (u[0] - u[2]) + p * r * (In[0] - In[2])),0.1e1 / In[2] * (p * q * (In[0] - In[1]) + (u[0] - u[1] + u[2] - u[3]) * kM)])
    return f


def quatQuadJac(x, u, m, g, kF, kM, L, In):
    """Return Jx and Ju, in quaternion case"""
    qw, qx, qy, qz = x[3], x[4], x[5], x[6]
    xd, yd, zd = x[7], x[8], x[9]
    p, q, r = x[10], x[11], x[12]
    t1 = p / 2
    t2 = q / 2
    t3 = qx / 2
    t4 = qy / 2
    t5 = qz / 2
    t6 = r / 2
    t7 = qw / 2
    t8 = (u[0] + u[1] + u[2] + u[3]) * kF
    t9 = 0.1e1 / m
    t10 = 2 * t9
    t11 = t10 * qw * t8
    t12 = t10 * qx * t8
    t13 = t10 * qy * t8
    t14 = t10 * qz * t8
    t15 = t10 * (qw * qy + qx * qz) * kF
    t10 = t10 * (qw * qx - qy * qz) * kF
    t16 = t9 * (-2 * qx ** 2 - 2 * qy ** 2 + 1) * kF
    t9 = -4 * t9
    t17 = In[1] - In[2]
    t18 = 0.1e1 / In[0]
    t19 = t18 * L * kF
    t20 = -In[0] + In[2]
    t21 = 0.1e1 / In[1]
    t22 = t21 * L * kF
    t23 = -In[0] + In[1]
    t24 = 0.1e1 / In[2]
    t25 = t24 * kM
    cg0 = np.array([[0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],[0,0,0,0,t6,-t2,t1,0,0,0,t5,-t4,t3,0,0,0,0],[0,0,0,-t6,0,t1,t2,0,0,0,t4,t5,-t7,0,0,0,0],[0,0,0,t2,-t1,0,t6,0,0,0,-t3,t7,t5,0,0,0,0],[0,0,0,-t1,-t2,-t6,0,0,0,0,-t7,-t3,-t4,0,0,0,0],[0,0,0,t13,t14,t11,t12,0,0,0,0,0,0,t15,t15,t15,t15],[0,0,0,-t12,-t11,t14,t13,0,0,0,0,0,0,-t10,-t10,-t10,-t10],[0,0,0,0,t9 * qx * t8,t9 * qy * t8,0,0,0,0,0,0,0,t16,t16,t16,t16],[0,0,0,0,0,0,0,0,0,0,0,t18 * r * t17,t18 * q * t17,0,t19,0,-t19],[0,0,0,0,0,0,0,0,0,0,t21 * r * t20,0,t21 * p * t20,-t22,0,t22,0],[0,0,0,0,0,0,0,0,0,0,-t24 * q * t23,-t24 * p * t23,0,t25,-t25,t25,-t25]])
    Jx, Ju = cg0[:, :13], cg0[:, 13:]
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
