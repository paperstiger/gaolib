#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Gao Tang <gt70@duke.edu>
#
# Distributed under terms of the MIT license.
from .gradient import gradNorm, verboseGradNorm, finiteDiff
from .local import Query
from .stat import stdify, destdify, l1loss, getPCA, getStandardData
from .extnp import blockIndex
from .local import get_affinity_matrix_xy as getAffinityXY
from .local import get_nn_index as getNNIndex
from .local import get_affinity_matrix_xy
from .local import get_nn_index
