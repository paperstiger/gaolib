#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Gao Tang <gt70@duke.edu>
#
# Distributed under terms of the MIT license.

"""
argument.py

Easy arguments
"""
import argparse
import re
import warnings
from .stringio import numRe, getNumber


_find_letter_pattern_ = re.compile(r'^([a-zA-Z]+)')  # this find first word
_has_number_pattern_ = re.compile(r'\d+')  # this detects if there is number
_letter_int_pattern_ = re.compile(r'^[a-zA-Z]+([-+]?[\d]+)$')  # this detects if it is word+int pattern


def getOnOffArgs(*args):
    """Get on-off arguments.

    This is a very useful function for defining problem arguments

    It supports the following types of arguments

    str: a pure string, it create a argument str and store bool, if str is in inputs, it is set to True
    str+space+str: defining string arguments, the first part is argument name, second is default value
    str+int: defining integer arguments
    str+float: defining float arguments

    A helper argument can always be added if arg is a tuple of two
    """
    parser = argparse.ArgumentParser()
    for arg_ in args:
        if isinstance(arg_, tuple):
            arg = arg_[0]
            helper = arg_[1]
        else:
            arg = arg_
            helper = None
        assert isinstance(arg, str)
        number_match = _has_number_pattern_.search(arg)
        if not number_match:  # only has word
            if ' ' not in arg:
                parser.add_argument('-%s' % arg, action='store_true', default=False, help=helper)
            else:
                s = arg.split(' ')
                parser.add_argument('-%s' % s[0], type=str, default=s[1], help=helper)
        else:  # has number
            # find the first word
            word_match = _find_letter_pattern_.search(arg)
            if word_match is None:
                warnings.warn("pattern %s does not match things" % arg)
                continue
            word = word_match.group(0)
            letter_int_match = _letter_int_pattern_.search(arg)
            if letter_int_match:  # word + int pattern
                integer = int(letter_int_match.groups()[0])
                parser.add_argument('-%s' % word, type=int, default=integer, help=helper)
            else:  # we assume a float
                floatnum = float(getNumber(arg)[0])
                parser.add_argument('-%s' % word, type=float, default=floatnum, help=helper)
    return parser.parse_args()
