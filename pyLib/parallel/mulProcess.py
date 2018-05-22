#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Gao Tang <gt70@duke.edu>
#
# Distributed under terms of the MIT license.

"""
mulProcess.py

My implementation of a simple class that supports multiprocessing.
We are doing function level
"""
import numpy as np
from multiprocessing import Manager, Process, RawArray
import copy
from functools import partial
import time
import pdb
import sys


class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child

    """
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin


class sharedNumpy(object):
    """A wrapper for shared numpy array"""
    def __init__(self, A):
        self.dtype = A.dtype
        self.shape = A.shape
        self.dtype = np.sctype2char(self.dtype)
        self.arr = RawArray(self.dtype, A.size)
        memoryview(self.arr)[:] = A.ravel()

    def numpy(self):  # Pytorch naming convention
        return np.reshape(np.frombuffer(self.arr, dtype=self.dtype), self.shape)


class monteCarlo(object):
    """A class that supports monte-carlo style simulation"""
    def __init__(self, f, *args, **kwargs):
        self.fun = f
        self.args = args
        self.kwargs = kwargs
        self.rst = None

    def __call__(self, i=None, dct=None):
        self.rst = self.fun(*self.args, **self.kwargs)
        # ForkedPdb().set_trace()
        if i is not None and dct is not None:
            dct[i] = self.rst
        else:
            return self.rst


class mulProcess(object):
    """A class for multi processing"""
    def __init__(self, fun, lsti, nProcess=None, *args, **kwargs):
        if nProcess is not None:
            assert len(lsti) == nProcess
        else:
            nProcess = len(lsti)
        self.MCs = []
        for i in range(nProcess):
            argsin = [lsti[i]]
            if args is not None:
                argsin.extend(args)
            self.MCs.append(monteCarlo(fun, *argsin, **kwargs))
        manager = Manager()
        self.return_dict = manager.dict()
        self.nProcess = nProcess

    def run(self, **kwargs):
        allproc = [Process(target=mc, args=(i, self.return_dict)) for i, mc in enumerate(self.MCs)]
        for proc in allproc:
            proc.start()
            if 'wait' in kwargs:
                time.sleep(kwargs['wait'])
        for proc in allproc:
            proc.join()
        results = []
        for i in range(self.nProcess):
            try:
                toappend = self.return_dict[i]
                results.append(toappend)
            except:
                print('Error occurs at %d' % i)
        return results


def getTaskSplit(num, nProcess):
    """Return a split of task."""
    tmp = np.linspace(0, num, nProcess + 1, dtype=int)
    return [(tmp[i], tmp[i + 1]) for i in range(nProcess)]
