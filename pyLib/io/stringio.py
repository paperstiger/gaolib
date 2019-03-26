#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Gao Tang <gt70@duke.edu>
#
# Distributed under terms of the MIT license.

"""
stringio.py

Provides subroutines to extract numbers from string
"""
import re


numRe = re.compile(r"[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?")


def getNumber(string, mapfun=None):
    """Parse all numbers from a string"""
    if mapfun is None:
        return numRe.findall(string)
    else:
        return map(mapfun, numRe.findall(string))


def joinNumber(arr, sign='_'):
    """Join a list of numbers into a single string."""
    return sign.join(map(str, arr))
