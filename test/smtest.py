#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Gao Tang <gt70@duke.edu>
#
# Distributed under terms of the MIT license.

"""
smtest.py

test shared memory module
"""
import sys, os, time
import numpy as np
import SharedArray as sa
import pyLib.io.sharedmemory as sm
from copy import deepcopy


def main():
    sm.clear()
    # add an array
    a = np.array([1, 0])
    sm.add(a, 'a')
    print(sm.get('a'))
    # add a dict by it
    d = {'b': np.random.random(3), 'c': np.random.random(2)}
    sm.add(d)
    sm.list()
    print(sm.get('b'))
    print(sm.get('c'))
    # add a dict with name attached
    e = deepcopy(d)
    sm.add(e, 'd')
    sm.list()
    print(sm.get('d'))


if __name__ == '__main__':
    main()
