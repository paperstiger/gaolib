#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Gao Tang <gt70@duke.edu>
#
# Distributed under terms of the MIT license.

"""
In this file, we have a few function that can change the color of printed things.
"""
from __future__ import print_function


def print_green(*args):
    print("\033[1;32m", sep='', *args)
    print("\033[0m", end='')


def print_yellow(*args):
    print("\033[1;33m", sep='', *args)
    print("\033[0m", end ='')


def print_purple(*args):
    print("\033[1;34m", sep='', *args)
    print("\033[0m", end='')


def print_red(*args):
    print("\033[1;35m", sep='', *args)
    print("\033[0m", end='')


def print_cyan(*args):
    print("\033[1;36m", sep='', *args)
    print("\033[0m", end='')


def print_gray(*args):
    print("\033[1;37m", sep='', *args)
    print("\033[0m", end='')


if __name__ == '__main__':
    print_green('haha')
