#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Gao Tang <gt70@duke.edu>
#
# Distributed under terms of the MIT license.
from .gradient import gradNorm, verboseGradNorm, finiteDiff
from .local import Query
from .stat import stdify, destdify, l1loss
