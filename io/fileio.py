#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Gao Tang <gt70@duke.edu>
#
# Distributed under terms of the MIT license.

"""
fileio.py

Useful functions for file io, especially dict of dict type.
This type is heavily used for clustered training.
"""
import numpy as np
try:
    import cPickle as pkl
except:
    import pickle as pickle


def ddctParse(fnm):
    """Parse a file that is dict of dict.
    I wish this is faster than pickle
    """
    if '.pkl' in fnm:
        with open(fnm, 'rb') as f:
            return pkl.load(f)
    tmp = np.load(fnm)
    keys = tmp.keys()
    try:
        rst = dict()
        for key in keys:
            rst[key] = tmp[key].item()
    except:
        rst = dict()
        for key in keys:
            rst[key] = tmp[key]
    return rst


def ddctSave(fnm, arr, pklmode=False):
    """Save a dict of dict"""
    assert isinstance(arr, dict)
    if not pklmode:
        np.savez(fnm, **arr)
    else:
        with open(fnm, 'wb') as f:
            pkl.dump(arr, f)
