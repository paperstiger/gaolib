#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Gao Tang <gt70@duke.edu>
#
# Distributed under terms of the MIT license.

"""
argtest.py

Test getOnOffArgs module, we add numbers
"""
from pyLib.io import getOnOffArgs


def main():
    args = getOnOffArgs('a', 'b1', 'c-1', 'd-0.5', 'e1e4', 'f-2.3e-5')
    print(args)


if __name__ == '__main__':
    main()
