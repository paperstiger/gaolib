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
