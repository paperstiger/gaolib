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
from multiprocessing import Manager, Process
import copy
from functools import partial


class monteCarlo(object):
    """A class that supports monte-carlo style simulation"""
    def __init__(self, f, *args, **kwargs):
        self.fun = f
        self.args = args
        self.kwargs = kwargs
        self.rst = None

    def __call__(self, i=None, dct=None):
        self.rst = self.fun(*self.args, **self.kwargs)
        if i is not None and dct is not None:
            dct[i] = self.rst


class mulProcess(object):
    """A class for multi processing"""
    def __init__(self, fun, lsti, nProcess, *args, **kwargs):
        assert len(lsti) == nProcess
        self.MCs = []
        for i in range(nProcess):
            argsin = [lsti[i]]
            if args is not None:
                argsin.extend(args)
            self.MCs.append(monteCarlo(fun, *argsin, **kwargs))
        manager = Manager()
        self.return_dict = manager.dict()
        self.nProcess = nProcess

    def run(self):
        allproc = [Process(target=mc, args=(i, self.return_dict)) for i, mc in enumerate(self.MCs)]
        for proc in allproc: proc.start()
        for proc in allproc: proc.join()
        results = [self.return_dict[i] for i in range(self.nProcess)]
        return results
