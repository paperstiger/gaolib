#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Gao Tang <gt70@duke.edu>
#
# Distributed under terms of the MIT license.

"""
test function in train module
"""
from __future__ import print_function, division
import sys, os, time
import numpy as np
import matplotlib.pyplot as plt

from pyLib.io import getArgs
from pyLib.train import GaoNetBN, trainOne, genTrainConfig


def main():
    net = GaoNetBN([2, 10, 2])
    net.eval()
    y = net.eval(np.random.random(2))
    print(y)
    # train a mapping with those variables
    net.train()
    N = 50
    xin = np.random.random((N, 2))
    y = np.c_[0.5*xin[:, 0], 0.6*xin[:, 1] + 0.4]
    data_dict = {'x': xin, 'y': y}
    config = genTrainConfig(network=net)
    trainOne(config, data_dict, net=net, scalex=False, scaley=False)
    net.printWeights()


if __name__ == '__main__':
    main()
