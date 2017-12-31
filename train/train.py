#! /usr/bin/env python
"""
Train a neural network to approximate the function, which is 
y = f(x, o)
where x is the collision trajectory, o is obstacle, and y is another collision-free traj
We hope we can directly learn the map so a good initial guess can be provided for later use
"""
import os, sys, time
import numpy as np
import matplotlib.pyplot as plt
import operator
import cPickle as pickle
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset
from torchUtil import GaoNet, plotError
from dataLoader import dataLoader
from tensorboardX import SummaryWriter


class trainer(object):
    """
    A base class for trainer.
    The user has to provide a trainSet and testSet, a net, a fun for loss, and other setting
    """
    def __init__(self, net, trainLder, testLder, trainloss, testloss=None, gtfun=operator.gt, **kwargs):
        assert isinstance(net, nn.Module)
        assert isinstance(trainLder, dataLoader)
        assert isinstance(testLder, dataLoader)
        self.net = net
        self.loss = trainloss
        if testloss is not None:
            self.testloss = testloss
        else:
            self.testloss = trainloss
        self.gtfun = gtfun  # in case we do classification
        self.trainLder = trainLder
        self.testLder = testLder
        # for data loading
        self.xname = kwargs.get('xname', 'X')
        self.yname = kwargs.get('yname', 'Y')
        # for training
        self.numEpoch = kwargs.get('epoch', 10)
        self.lr = kwargs.get('lr', 1e-3)
        self.errorBackStep = kwargs.get('errorbackstep', 1000)
        self.recordFreq = kwargs.get('recordfreq', 10)
        # for periodic checking to avoid slow progress
        self.enableTimer = kwargs.get('enabletimer', 1)
        self.timeCheckFreq = kwargs.get('timecheckfreq', 60)  # check every 60 seconds
        self.progressFactor = kwargs.get('progressFactor', 0.95)
        # others such as cuda
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)

    def train(self, saveName=None):
        # record loss for both training and test set
        maxRecordNum = self.numEpoch * len(self.trainLder) // self.recordFreq
        curRecord = 0
        checkTestError = None  # this tuple contains index and value
        lastCheckValue = None
        testErrorBackStep = self.errorBackStep
        trainerror = np.zeros(maxRecordNum)
        testerror = []  # a list since it might contain several things
        curIter = 0
        trainEnd = ''
        if self.enableTimer:
            curtime = time.time()
            nextCheckTime = curtime + self.timeCheckFreq
            progressFactor = self.progressFactor

        for epoch in range(self.numEpoch):
            print '\nEntering Epoch %d ' % epoch
            for idx, batch_data in enumerate(self.trainLder):
                self.optimizer.zero_grad()
                feedy = Variable(batch_data[self.yname], requires_grad=False).float().cuda()
                feedx = Variable(batch_data[self.xname], requires_grad=False).float().cuda()
                predy = self.net(feedx)
                trainloss = self.loss(predy, feedy)
                trainloss.backward()
                self.optimizer.step()
                if (curIter + 1) % self.recordFreq == 0:
                    errtrain = trainloss.cpu().data.numpy()[0]
                    trainerror[curRecord] = errtrain
                    # get test error
                    errtest = self.getTestLoss()
                    testerror.append(errtest)
                    # check if we have decreasing test error
                    if checkTestError is None:
                        checkTestError = (curIter, errtest)
                    elif self.gtfun(checkTestError[1], errtest):
                        checkTestError = (curIter, errtest)
                    curtime = time.time()
                    # check if progress is small
                    if curtime > nextCheckTime:
                        if lastCheckValue is not None and self.gtfun(checkTestError[1], progressFactor * lastCheckValue):
                            print('Break since we are making slow progress')
                            trainEnd = 'slow progress'
                            break
                        else:
                            lastCheckValue = checkTestError[1]
                            nextCheckTime = curtime + self.timeCheckFreq
                    # TODO: enable tensorboard record
                    """
                    if RECORD:
                        writer.add_scalar('data/trainloss', trainerror[curRecord], curIter)
                        if n
                        writer.add_scalar('data/testloss', testerror[curRecord], curIter)
                    """
                    curRecord += 1
                    print('\rstep {}, train error {}, test error {}'.format(curIter, errtrain, errtest))
                    # check if we have to stop now
                    if testErrorBackStep > 0 and curIter > checkTestError[0] + testErrorBackStep:
                        trainEnd = 'no improve'
                        break
                curIter += 1
            # check trainEnd
            if trainEnd == 'slow progress':
                print('Last value is %f ' % checkTestError[1])
                break
            elif trainEnd == 'no improve':
                print '\nTraining terminated since no progress is made within %d iter' % (testErrorBackStep)
                break
        # TODO: enable tensorboard
        """
        if RECORD:
            writer.close()
        """
        trainerror = trainerror[:curRecord]
        testerror = testerror[:curRecord]
        # TODO: enable retrain
        """
        if retrain:
            trainerror = np.concatenate((prevTrainError, trainerror))
            testerror = np.concatenate((prevTrainError, testerror))
        """
        # TODO: enable saving training progress to disk
        """
        # we store the training process
        pngname = getFileName(baseName, 'train')
        txtname = os.path.join(glbVar.PRJPATH, 'Script/models/record.txt')
        plotError(trainerror, testerror, recordFreq, pngname, False, show=False, txtname=txtname, mdlname=baseName)
        """
        # we save model and training errors
        if saveName is not None:
            model = {'model': self.net, 'trainerror': trainerror, 'testerror': testerror}
            if hasattr(self.trainLder, 'xmean'):
                model['xScale'] = [self.trainLder.xmean, self.trainLder.xstd]
            if hasattr(self.trainLder, 'ymean'):
                model['yScale'] = [self.trainLder.ymean, self.trainLder.ystd]
            with open(saveName, 'wb') as f:
                pickle.dump(model, f)

    def getTestLoss(self):
        dL = self.testLder
        rst = []
        namex, namey = self.xname, self.yname
        nData = self.testLder.getNumData()
        for idx, batch_data in enumerate(dL):
            feedy = Variable(batch_data[namey], volatile=True).cuda()
            feedx = Variable(batch_data[namex], volatile=True).cuda()
            predy = self.net(feedx)
            testloss = self.loss(predy, feedy)
            testloss = testloss.cpu().data.numpy()
            rst.append(testloss * len(batch_data[namex]))
        rst = np.array(rst)
        return np.sum(rst, axis=0)/nData  # we get a row vector, I believe


def getFileName(basename, prename, obj):
    if obj == 'log':
        return os.path.join(prename, 'runs', basename)
    elif obj == 'model':
        return os.path.join(prename, 'models', basename+'.pkl')
    elif obj == 'train':
        return os.path.join(prename, 'gallery', basename+'.png')
