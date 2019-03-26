#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Gao Tang <gt70@duke.edu>
#
# Distributed under terms of the MIT license.

"""
printio.py

Control print
"""
import sys
import os


class Mute(list):
    """Capture std.output."""
    def __init__(self, mute=True):
        self.mute = mute

    def __enter__(self):
        if self.mute:
            devnull = open(os.devnull, 'w')
            self.old_std = os.dup(sys.stdout.fileno())
            os.dup2(devnull.fileno(), 1)
            self.devnull = devnull

    def __exit__(self, *args):
        if self.mute:
            os.dup2(self.old_std, 1)
            self.devnull.close()
