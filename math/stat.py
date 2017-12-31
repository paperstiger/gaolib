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


def stdify(x, mean, std):
    """
    Given x, mean, and std, return the standarized x
    """
    return (x - mean) / std

def destdify(x, mean, std):
    """
    Recover to original x
    """
    return mean + std * x
