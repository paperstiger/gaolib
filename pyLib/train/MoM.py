#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Gao Tang <gt70@duke.edu>
#
# Distributed under terms of the MIT license.

"""
MoM.py

Utility functions for mixture of models
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable, grad
from collections import OrderedDict
import sys, os, time
import numpy as np
import matplotlib.pyplot as plt
import logging
from torchUtil import GaoNet, modelLoaderV2
from ..math.stat import destdify, stdify
import re
import cPickle as pickle


logger = logging.getLogger(__name__)
nameGet = re.compile(r'\/((?:.(?!\/))+)\..*$')
name2Get = re.compile(r'(.*)_at')


class MoMNet(nn.Module):
    """A network architecture that my mom told me"""
    def __init__(self, clus, mdls):
        """Can be initialized using either models or layer neurons"""
        super(MoMNet, self).__init__()
        if isinstance(clus, list) and isinstance(mdls, list):
            assert len(clus) >= 3
            assert clus[-1] == len(mdls)
            for mdl in mdls:
                assert len(mdl) >= 3
            self.clus = GaoNet(clus)  # a net of cluster
            self.mdls = [GaoNet(mdl) for mdl in mdls]  # a net of regressor
            self.useScale = False
        elif isinstance(clus, (str, unicode)):
            assert isinstance(mdls, list)
            for mdl in mdls:
                assert isinstance(mdl, (unicode, str))
            self.clus, self.clusScale, _ = modelLoaderV2(clus, True)
            self.mdls = []
            self.mdlsScale = []
            for mdl in mdls:
                tmpmdl, xScale, yScale = modelLoaderV2(mdl, True)
                self.mdls.append(tmpmdl)
                self.mdlsScale.append([xScale, yScale])
            self.useScale = True
            self.clusName = nameGet.findall(clus)[0]
            self.mdlsName = []
            for nm in mdls:
                nm0 = nameGet.findall(nm)[0]
                if '_at' in nm0:
                    self.mdlsName.append(name2Get.findall(nm0)[0])
                else:
                    self.mdlsName.append(nm0)
        self.main = nn.Sequential(OrderedDict([(str(i), lyr) for i, lyr in enumerate(self.mdls)]))
        self.EPS = 1.0
        self.argmax = True

    @property
    def numModel(self):
        return len(self.mdls)

    def parameters(self):
        self.clusPara = []
        self.clusPara.extend(self.clus.parameters())
        self.mdlsPara = []
        [self.mdlsPara.extend(mdl.parameters()) for mdl in self.mdls]
        return self.clusPara + self.mdlsPara

    def cuda(self):
        self.clus.cuda()
        [mdl.cuda() for mdl in self.mdls]
        if self.useScale:
            [x.cuda() for x in self.clusScale]
            [x.cuda() for z in self.mdlsScale for y in z for x in y]

    def getClusY(self, x):
        """Take x as input, return prediction of y from classifier"""
        xdim = 100
        if isinstance(x, np.ndarray):
            xdim = x.ndim
            x = Variable(torch.from_numpy(x).float(), volatile=True).cuda()
        if self.useScale:
            classXmean, classXstd = self.clusScale
            clsx = (x - classXmean) / classXstd  # This operation has been validated
        else:
            clsx = x
        clusy = self.clus(clsx)  # get actual prediction, in y, not prob
        clusy = clusy.cpu().data.numpy()
        if xdim == 1:
            clusy = np.squeeze(clusy, axis=0)
        return clusy

    def getPredY(self, x):
        """Take x as input, return predicted y, I do not take care of xScale or yScale since I do not know it"""
        if isinstance(x, np.ndarray):
            xdim = x.ndim  # detect which case to use
            if xdim == 1:
                x = np.expand_dims(x, axis=0)  # make it 1 by X
            feedx = Variable(torch.from_numpy(x).float(), volatile=True).cuda()
        else:
            feedx = x
        predy = self.forward(feedx).cpu().data.numpy()
        if isinstance(x, np.ndarray):
            if xdim == 1:
                predy = np.squeeze(predy, axis=0)
        return predy

    def getPredYi(self, x, i):
        """Take x as input, use the i-th regressor to predict y."""
        if isinstance(x, np.ndarray):
            xdim = x.ndim
            if xdim == 1:
                x = np.expand_dims(x, axis=0)
            feedx = Variable(torch.from_numpy(x).float(), volatile=True).cuda()
        else:
            feedx = x
        if self.useScale:
            regXmean, regXstd = self.mdlsScale[i][0]
            feedx = (feedx - regXmean) / regXstd
        predyi = self.mdls[i](feedx)
        if self.useScale:
            regYmean, regYstd = self.mdlsScale[i][1]
            predyi = predyi * regYstd + regYmean
        predyi = predyi.cpu().data.numpy()
        if isinstance(x, np.ndarray):
            if xdim == 1:
                predyi = np.squeeze(predyi, axis=0)
        return predyi

    def forward(self, x):  # default setting is here
        # scale x to required scale
        if self.useScale:
            classXmean, classXstd = self.clusScale
            clsx = (x - classXmean) / classXstd  # This operation has been validated
        else:
            clsx = x
        clusy = self.clus(clsx)  # get actual prediction, in y, not prob
        if not self.argmax:
            clusy /= self.EPS
            lblpred = nn.functional.softmax(clusy, dim=1)
            lblpred = lblpred.unsqueeze(2)
            vecui = []
            for i, net in enumerate(self.mdls):
                if self.useScale:
                    regXmean, regXstd = self.mdlsScale[i][0]
                    regYmean, regYstd = self.mdlsScale[i][1]
                    regx = (x - regXmean) / regXstd
                else:
                    regx = x
                predy = net(regx)
                if self.useScale:
                    y = predy * regYstd + regYmean
                else:
                    y = predy
                vecui.append(y)
            allui = torch.stack(tuple(vecui), 2)
            outy = torch.bmm(allui, lblpred)
            outy = outy.squeeze()
        else:
            lblval, lblpred = torch.max(clusy, dim=1)
            lblpred = lblpred.cpu().data.numpy()
            nbatch = x.size()[0]
            outy = None
            for i, net in enumerate(self.mdls):
                ind = np.where(lblpred == i)[0]
                if len(ind) > 0:
                    ind = Variable(torch.from_numpy(ind)).cuda()
                    regx = torch.index_select(x, 0, ind)
                    if self.useScale:
                        regXmean, regXstd = self.mdlsScale[i][0]
                        regYmean, regYstd = self.mdlsScale[i][1]
                        regx = (regx - regXmean) / regXstd
                        y = net(regx)
                    else:
                        y = net(regx)
                    if self.useScale:
                        y = y * regYstd + regYmean
                    if outy is None:
                        ny = y.size()[1]
                        outy = Variable(torch.zeros((nbatch, ny)).cuda())
                    outy[ind] = y
        return outy

    def getWeights(self):
        vw, vb = [], []
        for m in self.modules():
            if isinstance(m, nn.Linear):
                vw.append(m.weight.data.numpy())
                vb.append(m.bias.data.numpy())
        return {'w': vw, 'b': vb}

    def __str__(self):
        if not self.useScale:
            nms = []
            nms.append('clus_%s_model' % str(self.clus))
            [nms.append('-%s' % str(mdl)) for mdl in self.mdls]
            nm = ''.join(nms)
            return nm
        else:
            return self.clusName + '_' + '-'.join(self.mdlsName)


def momLoader(model, withclus=False, argmax=False):
    """Given a model name, usually a pickle file, load the model, create a function.
    It is loading mom, so it returns both predY and y, the first is used to recover classification
    """
    try:
        with open(model, 'rb') as f:
            tmp = pickle.load(f)
        mdl = tmp['model']
    except:
        tmp = torch.load(model)
        mdl = tmp['model']
    mdl.cuda()
    if argmax:
        mdl.argmax = True
    xScale = tmp.get('xScale', None)
    yScale = tmp.get('yScale', None)

    # create the function
    def fun(x, batch_size=-1):
        '''given x, return y, x can be batch or single.'''
        if batch_size == -1:
            xdim = x.ndim  # detect which case to use
            if xdim == 1:
                x = np.expand_dims(x, axis=0)  # make it 1 by X
            if xScale is not None:
                x = stdify(x, xScale[0], xScale[1])
            # convert and feed
            feedx = Variable(torch.from_numpy(x.astype(np.float32)), volatile=True).cuda()
            clusy = mdl.getClusY(feedx)
            predy = mdl(feedx).cpu().data.numpy()
            if yScale is not None:
                predy = destdify(predy, yScale[0], yScale[1])
            if xdim == 1 and predy.ndim == 2:
                predy = np.squeeze(predy, axis=0)
            if withclus:
                return clusy, predy
            else:
                return predy
        else:  # we have to do it in batch to save memory or other stuff
            assert batch_size > 0
            nData = len(x)
            nmodel = len(mdl.mdls)
            if isinstance(batch_size, float) and batch_size < 1.0:
                batchsize = int(batch_size * nData)
            else:
                batchsize = batch_size
            # detect size of output
            if yScale is not None:
                dimy = yScale[0].shape[1]  # since it is 1 by dimy
            else:
                feedx = Variable(torch.from_numpy(x[:1]).float(), volatile=True)
                if cuda:
                    feedx = feedx.cuda()
                predy = mdl(feedx).cpu().data.numpy()
                dimy = predy.shape[1]
            # allocate space
            predy = np.zeros((nData, dimy), dtype=np.float32)
            if withclus:
                clusy = np.zeros((nData, nmodel), dtype=np.int)
            loopN = nData // batchsize + 1
            if nData % batchsize == 0:
                loopN -= 1
            for i in range(loopN):
                ind0, indf = batchsize*i, batchsize*(i + 1)
                if indf > nData:
                    indf = nData
                if withclus:
                    tmpclusy, tmpy = fun(x[ind0:indf])
                    clusy[ind0:indf] = tmpclusy
                else:
                    tmpy = fun(x[ind0:indf])
                    predy[ind0:indf] = tmpy
            if withclus:
                return clusy, predy
            else:
                return predy

    return fun
