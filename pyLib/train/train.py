#! /usr/bin/env python
"""
Train a neural network to approximate the function, which is 
y = f(x, o)
where x is the collision trajectory, o is obstacle, and y is another collision-free traj
We hope we can directly learn the map so a good initial guess can be provided for later use
"""
import os, sys, time, datetime
import numpy as np
import matplotlib.pyplot as plt
import operator
try:
    import cPickle as pickle
except:
    import pickle
import glob
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset
from .torchUtil import GaoNet, GaoNetBN, plotError, recordStep0
from .dataLoader import dataLoader, keyFactory, labelFactory, subFactory, unaryKeyFactory
from .torchUtil import modelLoader
# from tensorboardX import SummaryWriter


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
        if hasattr(trainLder, '_xname'):
            self.xname = trainLder._xname
        if hasattr(trainLder, '_yname'):
            self.yname = trainLder._yname
        # for training
        self.numEpoch = kwargs.get('epoch', 10)
        self.lr = kwargs.get('lr', 1e-3)
        self.errorBackStep = kwargs.get('errorbackstep', 1000)
        self.epochBackStep = kwargs.get('epochbackstep', 10)
        self.recordFreq = kwargs.get('recordfreq', 10)
        # for periodic checking to avoid slow progress
        self.enableTimer = kwargs.get('enabletimer', 1)
        self.timeCheckFreq = kwargs.get('timecheckfreq', 60)  # check every 60 seconds
        self.progressFactor = kwargs.get('progressFactor', 0.95)
        self.unary = kwargs.get('unary', 0)  # by default not unary
        self.txtname = kwargs.get('txtname', 'models/record.txt')
        self.overWriteModel = kwargs.get('overwrite', True)
        self.epochSaveFreq = 100
        # others such as cuda
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)

    def getTestError(self):
        """Evaluate current model using test set, report average error"""
        return self.getTestLoss()

    def setEpochSaveFreq(self, freq):
        self.epochSaveFreq = freq

    def getTrainError(self):
        return self.getTestLoss(self.trainLder)

    def modifySaveName(self, saveName, overwrite=True):
        """Optionally change saveName to avoid overwriting of models"""
        if saveName is None:
            return saveName
        abspath = os.path.abspath(saveName)
        dir_name = os.path.dirname(abspath)
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        if not overwrite:
            files = glob.glob('%s/*' % dir_name)
            left_files = filter(lambda x: x.startswith(abspath), files)
            if len(left_files) > 0:
                saveName += '.%d' % len(left_files)
        return saveName

    def train_epoch(self, saveName=None):
        """Train the model epoch by epoch so it is slightly different from train.

        I remove many arguments since they are not that useful.
        I will check training and test error at each epoch
        """
        # record loss for both training and test set
        maxRecordNum = self.numEpoch
        checkTestError = None  # this tuple contains index and value
        epochBackStep = self.epochBackStep
        trainerror = np.zeros(maxRecordNum)
        testerror = []  # a list since it might contain several things
        trainEnd = ''
        saveName = self.modifySaveName(saveName, self.overWriteModel)

        # record at entry
        testerror0 = self.getTestLoss()
        train_size = self.trainLder.numData

        # begin training
        for epoch in range(self.numEpoch):
            train_loss_sum = 0
            self.net.train()
            for idx, batch_data in enumerate(self.trainLder):
                if self.unary == 0:
                    if batch_data[self.xname].shape[0] == 1:
                        continue
                    feedy = Variable(batch_data[self.yname], requires_grad=False).cuda()
                    feedx = Variable(batch_data[self.xname], requires_grad=False).cuda()
                else:
                    if batch_data.shape[0] == 1:
                        continue
                    feedy = Variable(batch_data, requires_grad=False).float().cuda()
                    feedx = feedy
                self.optimizer.zero_grad()
                predy = self.net(feedx)
                trainloss = self.loss(predy, feedy)
                trainloss.backward()
                train_loss_sum += trainloss.cpu().data.numpy() * len(feedx)
                self.optimizer.step()
            # record average train loss
            trainerror[epoch] = train_loss_sum / train_size
            # evaluate on test set
            errtest = self.getTestLoss()
            testerror.append(errtest)
            print('Epoch ', epoch, 'train loss ', trainerror[epoch], 'test loss ', testerror[-1])
            # check if we have decreasing test error
            if checkTestError is None:
                checkTestError = (epoch, errtest)
            elif self.gtfun(checkTestError[1], errtest):
                checkTestError = (epoch, errtest)
            elif epoch > checkTestError[0] + epochBackStep:
                trainEnd = 'no improve'
                break
            # optionally save model
            if epoch % self.epochSaveFreq == self.epochSaveFreq - 1:
                self.save(saveName, False)

        # check if we truly exit in case numEpoch is not high enough
        if trainEnd == 'no improve':
            print('\nTraining terminated since no progress is made within %d epoch' % (epochBackStep))
        # truncate
        trainerror = trainerror[:epoch]
        testerror = testerror[:epoch]

        # output
        with open(self.txtname, 'a') as f:
            f.write('%s\n' % datetime.datetime.now())
            recordStep0(testerror0, saveName, self.txtname)
            plotError(trainerror, testerror, 1, None, False, False, txtname=self.txtname, mdlname=saveName)
        self.save(saveName, True, trainerror, testerror)

    def save(self, saveName, final=False, trainerror=None, testerror=None):
        # we save model and training errors
        if saveName is not None:
            if final:
                self.net.cpu()
            if trainerror is not None and testerror is not None:
                model = {'model': self.net, 'trainerror': trainerror, 'testerror': testerror}
            else:
                model = {'model': self.net}
            if hasattr(self.trainLder, 'xmean'):
                model['xScale'] = [self.trainLder.xmean, self.trainLder.xstd]
            if hasattr(self.trainLder, 'ymean'):
                model['yScale'] = [self.trainLder.ymean, self.trainLder.ystd]
            torch.save(model, saveName)

    def train(self, saveName=None, threshold=None, additional=None, ptmode=True, statedict=False):
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
        saveName = self.modifySaveName(saveName, self.overWriteModel)
        if self.enableTimer:
            curtime = time.time()
            nextCheckTime = curtime + self.timeCheckFreq
            progressFactor = self.progressFactor

        # record at entry
        testerror0 = self.getTestLoss()

        # begin training
        for epoch in range(self.numEpoch):
            print('\nEntering Epoch %d ' % epoch)
            for idx, batch_data in enumerate(self.trainLder):
                self.optimizer.zero_grad()
                if self.unary == 0:
                    feedy = Variable(batch_data[self.yname], requires_grad=False).cuda()
                    feedx = Variable(batch_data[self.xname], requires_grad=False).cuda()
                else:
                    feedy = Variable(batch_data, requires_grad=False).float().cuda()
                    feedx = feedy
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
                recordStep0(testerror0, saveName, self.txtname)
                plotError(trainerror, testerror, self.recordFreq, None, False, False, txtname=self.txtname, mdlname=saveName)
            except:
                print('error occurs when trying to record training progress')
                pass
        # we save model and training errors
        if saveName is not None:
            self.net.cpu()
            if statedict:
                model = {'model': self.net.state_dict(), 'trainerror': trainerror, 'testerror': testerror}
            else:
                model = {'model': self.net, 'trainerror': trainerror, 'testerror': testerror}
            if hasattr(self.trainLder, 'xmean'):
                model['xScale'] = [self.trainLder.xmean, self.trainLder.xstd]
            if hasattr(self.trainLder, 'ymean'):
                model['yScale'] = [self.trainLder.ymean, self.trainLder.ystd]
            if saveName.endswith('.pt'):
                ptmode = True
            if ptmode:
                newName = saveName.replace('.pkl', '.pt')
                torch.save(model, newName)
            else:
                with open(saveName, 'wb') as f:
                    pickle.dump(model, f)

    def getTestLoss(self, lder=None):
        self.net.eval()
        if lder is None:
            dL = self.testLder
        else:
            dL = lder
        rst = []
        namex, namey = self.xname, self.yname
        nData = dL.getNumData()
        for idx, batch_data in enumerate(dL):
            if self.unary == 0:
                lenData = len(batch_data[namex])
                feedy = Variable(batch_data[namey], volatile=True).cuda()
                feedx = Variable(batch_data[namex], volatile=True).cuda()
            else:
                lenData = len(batch_data)
                feedx = Variable(batch_data, volatile=True).cuda()
                feedy = feedx
            predy = self.net(feedx)
            testloss = self.testloss(predy, feedy)
            testloss = testloss.cpu().data.numpy()
            rst.append(testloss * lenData)
        rst = np.array(rst)
        return np.sum(rst, axis=0)/nData  # we get a row vector, I believe


def getFileName(basename, prename, obj):
    if obj == 'log':
        return os.path.join(prename, 'runs', basename)
    elif obj == 'model':
        return os.path.join(prename, 'models', basename+'.pkl')
    elif obj == 'train':
        return os.path.join(prename, 'gallery', basename+'.png')


def genFromDefaultConfig(**kwargs):
    """Generate a default configuration dict to avoid things.

    Available keywords include:
    network: a list of int, describing network structure
    lr: float, learning rate, default 1e-3
    epoch: int, number of learning epoch, default 1500
    batch_size: int, size of minibatch for sgd, default 64
    test_batch_size: int, size of minibatch for evaluating test set; -1 means all, default -1
    errorbackstep: int, if test error has no improvement within this steps, quit training, default 300
    epochbackstep: int, if test error has no improvement within this epoch, quit training, default 10
    trainsize: float, split of training and test set size, default 0.8
    recordfreq: int, frequency of recording of test errors, default 10
    namex: str, key that maps to input, default 'x'
    namey: str, key that maps to output, default 'y'
    ourdir: str, directory to store saved models, default 'models'
    outname: str, name for this particular model, default 'y_of_x.pt'
    dataname: str, name for the data, might be unnecessary, default 'data/data.npz'
    """
    defaultdict = {
                "network":  [
                            [1, 1, 1]
                            ],
                "neton": [1],
                "bn": True,  # enable batch normalization
                "lr": 1e-3,
                "epoch": 1500,
                "batch_size": 64,
                "test_batch_size": -1,
                "errorbackstep": 300,
                "epochbackstep": 8,
                "trainsize": 0.8,
                "recordfreq": 10,
                "savefreq": 10,
                "namex": "x",
                "namey": "y",
                "outdir": "models",
                "overwrite": True,
                "outname": "y_of_x.pt",
                'dataname': "data/data.npz"
                }
    if kwargs is not None:
        defaultdict.update(kwargs)
    if '.' not in defaultdict['outname']:
        defaultdict['outname'] = defaultdict['outname'] + '.pt'
    return defaultdict


def getModelPath(config):
    return os.path.join(config['outdir'], config['outname'])


def writeHeader(msg, fnm=None):
    """Write some message to models/record.txt"""
    if fnm is None:
        fnm = 'models/record.txt'
    try:
        with open(fnm, 'a') as f:
            f.write('\n%s\n' % datetime.datetime.now())
            f.write(msg)
            f.write('\n')
    except:
        print('error occurs when trying to write a message to %s' % fnm)


def trainOne(config, data, scale_back=False, seed=None, loss=None, is_reg_task=True, x0_name='x0', net=None, scalex=True, scaley=True):
    """Train a network and save it somewhere.

    It handles naive regression and classification task if GaoNet is used.
    Training hyperparameters are listed in config, which is a dictionary
    Data is a dictionary, for regression task, it has at least two entries, specified by config
    For classification task, it has keys like x0/x1, etc. each is another dict that has x0_name as key or ndarray

    Parameters
    ----------
    config : dict, containing training parameters
    data : dict, containing data for training
    scale_back : bool, if we calculate smooth L1 loss in original scale (default False)
    seed : int, seeder for random
    loss : callable, custom loss function
    is_reg_task : bool, if we are training a regression task instead of classification
    x0_name : str, the name of x in each cluster when classification task is performed
    net : a nn.Module object, if it is used, we do not construct GaoNet
    scalex : bool, if we standarize x before training
    scaley : bool, if we standarize y before training

    """
    trainsize = config['trainsize']
    epoch = config['epoch']
    lr = config['lr']
    recordfreq = config['recordfreq']
    errorbackstep = config['errorbackstep']
    epochbackstep = config['epochbackstep']
    over_write = config['overwrite']
    if is_reg_task:
        namex = config.get('namex', 'x')
        namey = config.get('namey', 'y')
        factory = keyFactory(data, namex, namey, scalex=scalex, scaley=scaley)
        factory.shuffle(seed)
    else:
        assert isinstance(data, dict)
        if 'x0' in data:  # for classifier training, you can either use dict of x0, x1, etc or dict of x, label, n_label
            nCluster = len(data)
            nameLblPair = [('x%d' % i, i) for i in range(nCluster)]
            if isinstance(data['x0'], dict):
                factory = labelFactory(data, nameLblPair, xfun=lambda x: x[x0_name], scalex=scalex)
            else:
                factory = labelFactory(data, nameLblPair, xfun=None, scalex=scalex)
        elif 'label' in data:
            factory = labelFactory(data, None, None, scalex=scalex)
        factory.shuffle(seed)
    trainSet = subFactory(factory, 0.0, trainsize)
    testSet = subFactory(factory, trainsize, 1.0)
    batch_size = config['batch_size']
    outname = os.path.join(config['outdir'], config['outname'])
    test_batch_size = config['test_batch_size']
    trainLder = dataLoader(trainSet, batch_size=batch_size, shuffle=False)
    testLder = dataLoader(testSet, batch_size=test_batch_size, shuffle=False)
    if net is None:
        network = _getNetwork(config)
        if config['bn']:
            net = GaoNetBN(network).cuda()
        else:
            net = GaoNet(network).cuda()
    else:
        net.cuda()

    if loss is not None:
        trner = trainer(net, trainLder, testLder, loss, epoch=epoch, lr=lr, recordfreq=recordfreq, epochbackstep=epochbackstep)
    else:
        if is_reg_task:
            ymean = Variable(torch.from_numpy(factory.ymean).float()).cuda()
            ystd = Variable(torch.from_numpy(factory.ystd).float()).cuda()
            l1loss = torch.nn.SmoothL1Loss(reduce=False)
            l1lossreduce = torch.nn.SmoothL1Loss(reduce=True)

            # define the loss function for trajectory regression uisng smooth l1 loss
            def testRegLoss(predy, feedy):
                y1 = predy * ystd
                y2 = feedy * ystd
                lossy1y2 = l1loss(y1, y2)
                loss = torch.mean(lossy1y2)
                return loss

            if scale_back:
                testloss = testRegLoss
            else:
                testloss = l1lossreduce

            trner = trainer(net, trainLder, testLder, testloss,
                            epoch=epoch, lr=lr, recordfreq=recordfreq, errorbackstep=errorbackstep, epochbackstep=epochbackstep)
        else:
            celoss = torch.nn.CrossEntropyLoss()

            # define the loss function for classification using cross entropy
            def testClassifyLoss(predy, feedy):
                """Return not only the loss but also accuracy."""
                loss1 = celoss(predy, feedy)
                maxs_x, indices_x = torch.max(predy, dim=1)  # find max along row
                correct = torch.eq(feedy, indices_x)
                correct = correct.float()
                accuracy = torch.mean(correct)
                return torch.cat((loss1, accuracy))

            def greater(arg1, arg2):
                return arg1[1] < arg2[1]

            trner = trainer(net, trainLder, testLder, celoss, testloss=testClassifyLoss, gtfun=greater,
                            epoch=epoch, lr=lr, recordfreq=recordfreq, errorbackstep=errorbackstep, epochbackstep=epochbackstep)

    trner.overWriteModel = over_write
    trner.setEpochSaveFreq(config['savefreq'])
    trner.train_epoch(outname)


def evalOne(config, x, cuda=False):
    """Load a model from config, evaluate the model on data x.

    :param config: dict, from calling genTrainConfig
    :param x: ndarray, input to the model
    :param cuda: bool, if we enable cudaify of the model
    """
    mdl_name = os.path.join(config['outdir'], config['outname'])
    mdlfun = modelLoader(mdl_name, cuda=cuda)
    return mdlfun(x)


def trainAutoEncoder(net, data, config, seed=1994, scale=False):
    """Train an autoencoder, an unsupervised learning model.

    The net has to be encoded by the user, if it is None, and config has network size we explore GaoNet to do this.

    Parameters
    ----------
    net : nn.Module object, the network to be trained
    config : dict, containing training parameters
    data : dict, containing data for training
    seed : int, seeder for random
    scale : if we scale the data to make it normal

    """
    assert isinstance(net, nn.Module)
    if net is None:
        network = _getNetwork(config)
        if config['bn']:
            net = GaoNetBN(network)
        else:
            net = GaoNet(network)
    net.cuda()
    trainsize = config['trainsize']
    epoch = config['epoch']
    lr = config['lr']
    recordfreq = config['recordfreq']
    errorbackstep = config['errorbackstep']
    epochbackstep = config['epochbackstep']
    factory = unaryKeyFactory({'x': data}, 'x', scalex=scale)
    factory.shuffle(seed)
    # split
    trainSet = subFactory(factory, 0.0, trainsize)
    testSet = subFactory(factory, trainsize, 1.0)
    # read batch
    batch_size = config['batch_size']
    outname = os.path.join(config['outdir'], config['outname'])
    test_batch_size = config['test_batch_size']
    trainLder = dataLoader(trainSet, batch_size=batch_size, shuffle=False)
    testLder = dataLoader(testSet, batch_size=test_batch_size, shuffle=False)

    l1loss = torch.nn.SmoothL1Loss()

    trner = trainer(net, trainLder, testLder, l1loss,
                    epoch=epoch, lr=lr, recordfreq=recordfreq, errorbackstep=errorbackstep, epochbackstep=epochbackstep, unary=1)
    trner.train_epoch(outname)


def _getNetwork(config):
    """Extract the network list from config.

    We note that for backward compatibility, we support [[x,x,x]] type.

    """
    if 'network' not in config:
        return None
    assert isinstance(config['network'], list)
    if isinstance(config['network'][0], list):
        network = config['network'][0]
    else:
        network = config['network']
    return network
