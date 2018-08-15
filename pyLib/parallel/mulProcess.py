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
        self.fun = fun
        self.nProcess = nProcess
        self.listi = lsti
        self.args = args
        self.kwargs = kwargs
        self.return_dict = manager.dict()
        self.enable_pid = False

    def enablePID(self):
        """Enable PID such that the id of the process will be passed automatically"""
        self.enable_pid = True

    def run(self, **kwargs):
        """Run the simulation in multiple processes.

        :param kwargs: key-word arguments, user can specify wait time by wait=0.1
        :return: a list of return values from each process
        """
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

    def debug(self):
        """Run the simulation in debug mode.

        In debug mode, we call the function directly in current process so pdb can add breakpoints.
        """
        self.fun(self.listi[0], *self.args, **self.kwargs)


def getTaskSplit(num, nProcess):
    """Return a split of task.

    :param num: int, total number of task
    :param nProcess: int, number of process
    """
    tmp = np.linspace(0, num, nProcess + 1, dtype=int)
    return [(tmp[i], tmp[i + 1]) for i in range(nProcess)]


def getSharedNumpy(*args):
    """Return the shared numpy wrapper for many numpy arrays.

    :param args: ndarrays
    :return: the shared numpy wrapper for each numpy array
    """
    if len(args) == 1:
        return sharedNumpy(args[0])
    return [sharedNumpy(arg) for arg in args]


def getNumpy(*args):
    """Return the numpy instance from shared numpy.

    :param args: sharedNumpy object
    :return: the numpy array
    """
    if len(args) == 1:
        return args[0].numpy()
    return [arg.numpy() for arg in args]
