#! /usr/bin/env python
"""
Train a neural network to approximate the function

The loss is scaled, the weight is passed in using vecKeyFactory
"""
import os, sys, time, datetime
import numpy as np
import matplotlib.pyplot as plt
import operator
try:
    import cPickle as pickle
except:
    import pickle
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset
from .torchUtil import GaoNet, plotError
from .dataLoader import dataLoader
from .train import trainer, getFileName


class weightTrainer(trainer):
    """
    A base class for trainer.
    The user has to provide a trainSet and testSet, a net, a fun for loss, and other setting
    """
    def __init__(self, net, trainLder, testLder, trainloss, testloss=None, gtfun=operator.gt, **kwargs):
        trainer.__init__(self, net, trainLder, testLder, trainloss, testloss, gtfun, **kwargs)
        self.wname = kwargs.get('wname', 'weight')

    def train(self, saveName=None, threshold=None, additional=None, ptmode=False):
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
            print('\nEntering Epoch %d ' % epoch)
            for idx, batch_data in enumerate(self.trainLder):
                self.optimizer.zero_grad()
                feedy = Variable(batch_data[self.yname], requires_grad=False).cuda()
                feedx = Variable(batch_data[self.xname], requires_grad=False).cuda()
                weight = Variable(batch_data[self.wname], requires_grad=False).cuda()
                # forward
                predy = self.net(feedx)
                trainloss = self.loss(predy, feedy, weight)  # trainloss accept three variables
                trainloss.backward()
                self.optimizer.step()
                if (curIter + 1) % self.recordFreq == 0:
                    errtrain = trainloss.cpu().data.numpy()[0]
                    trainerror[curRecord] = errtrain
                    # get test error
                    errtest = self.getTestLoss()
                    testerror.append(errtest)
                    curRecord += 1
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
                    print('\rstep {}, train error {}, test error {}'.format(curIter, errtrain, errtest))
                    # check if we have to stop now
                    if testErrorBackStep > 0 and curIter > checkTestError[0] + testErrorBackStep:
                        trainEnd = 'no improve'
                        break
                    # check if we have reached threshold
                    if threshold is not None and self.gtfun(threshold, checkTestError[1]):
                        trainEnd = 'threshold reached'
                        break
                curIter += 1
            # check trainEnd
            if trainEnd == 'slow progress':
                print('Last value is %f ' % checkTestError[1])
                break
            elif trainEnd == 'no improve':
                print('\nTraining terminated since no progress is made within %d iter' % (testErrorBackStep))
                break
            elif trainEnd == 'threshold reached':
                print('\nTraining terminated since threshold has been reached within %d iter' % (testErrorBackStep))
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
        if self.txtname is not None and saveName is not None:
            try:
                with open(self.txtname, 'a') as f:
                    f.write('%s\n' % datetime.datetime.now())
                    if additional is not None:
                        f.write('%s\n' % additional)
                plotError(trainerror, testerror, self.recordFreq, None, False, False, txtname=self.txtname, mdlname=saveName)
            except:
                print('error occurs when trying to record training progress')
                pass
        # we save model and training errors
        if saveName is not None:
            model = {'model': self.net, 'trainerror': trainerror, 'testerror': testerror}
            if hasattr(self.trainLder, 'xmean'):
                model['xScale'] = [self.trainLder.xmean, self.trainLder.xstd]
            if hasattr(self.trainLder, 'ymean'):
                model['yScale'] = [self.trainLder.ymean, self.trainLder.ystd]
            if ptmode:
                newName = saveName.replace('.pkl', '.pt')
                torch.save(model, newName)
            else:
                with open(saveName, 'wb') as f:
                    pickle.dump(model, f)

    def getTestLoss(self, lder=None):
        if lder is None:
            dL = self.testLder
        else:
            dL = lder
        rst = []
        namex, namey, namew = self.xname, self.yname, self.wname
        nData = self.testLder.getNumData()
        for idx, batch_data in enumerate(dL):
            lenData = len(batch_data[namex])
            feedy = Variable(batch_data[namey], volatile=True).cuda()
            feedx = Variable(batch_data[namex], volatile=True).cuda()
            weight = Variable(batch_data[namew], volatile=True).cuda()
            predy = self.net(feedx)
            testloss = self.testloss(predy, feedy, weight)
            testloss = testloss.cpu().data.numpy()
            rst.append(testloss * lenData)
        rst = np.array(rst)
        return np.sum(rst, axis=0)/nData  # we get a row vector, I believe
