#! /usr/bin/env python
"""
Utility functions combined with pytorch
Such as define a mlp network relying on a list of numbers
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable, grad
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
import pickle as pickle
from functools import partial
import os
from ..math.stat import destdify, stdify, getStandardData


class GaoNet(nn.Module):
    def __init__(self, lyrs, dropout=None, bn=False):
        """Constructor for GaoNet with backward compatibility. By default dropout or bn is disabled."""
        super(GaoNet, self).__init__()
        assert len(lyrs) >= 3
        self.lyrs = lyrs
        self.layers = []
        nHidden = len(lyrs) - 2
        lasthidden = lyrs[-2]
        outlayer = lyrs[-1]
        for i in range(nHidden):
            self.layers.append(nn.Linear(lyrs[i], lyrs[i+1]))
            self.layers.append(nn.LeakyReLU(0.2))
            if i > 0 and dropout is not None:
                self.layers.append(nn.Dropout(p=dropout))
            if bn:
                self.layers.append(nn.BatchNorm1d(lyrs[i + 1]))
        # final layer is linear output
        self.layers.append(nn.Linear(lasthidden, outlayer))
        # use the OrderedDict trick to assemble the system
        self.main = nn.Sequential(OrderedDict([(str(i), lyr) for i, lyr in enumerate(self.layers)]))

    def forward(self, x):
        return self.main(x)
        # out = x
        # for i, lyr in enumerate(self.layers):
        #     out = lyr(out)
        # return out

    def getWeights(self):
        vw, vb = [], []
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.BatchNorm1d, nn.BatchNorm2d)):
                vw.append(m.weight.data.numpy())
                vb.append(m.bias.data.numpy())
        return {'w': vw, 'b': vb}

    def eval(self, x=None):
        if x is None:
            return nn.Module.eval(self)
        assert isinstance(x, np.ndarray)
        ndim = x.ndim
        if ndim == 1:
            inx = Variable(torch.from_numpy(x[np.newaxis, :]).float(), volatile=True)
        else:
            inx = Variable(torch.from_numpy(x).float(), volatile=True)
        y = self.forward(inx).data.numpy()
        if ndim == 1:
            return np.squeeze(y, axis=0)
        else:
            return y

    def extendXYScale(self, xScale=None, yScale=None):
        """Extend scale we mean we extend this model such that if input removes this scaler, result does not change
        We do it by modifying weights. Although I do think linear layer should also work.
        """
        state_dct = self.state_dict()
        nlyr = len(self.layers)  # so I just use 0 and this number - 1
        if xScale is not None:
            w0 = state_dct['main.0.weight']
            b0 = state_dct['main.0.bias']
            xmean, xstd = xScale
            b0 -= torch.mv(w0, torch.from_numpy(np.squeeze(xmean/xstd, axis=0)).float())
            w0 /= torch.from_numpy(xstd).float()
        if yScale is not None:
            wf = state_dct['main.%d.weight' % (nlyr - 1)]
            bf = state_dct['main.%d.bias' % (nlyr - 1)]
            ymean, ystd = yScale
            wf *= torch.from_numpy(ystd.T).float()
            bf *= torch.from_numpy(np.squeeze(ystd, axis=0)).float()
            bf += torch.from_numpy(np.squeeze(ymean, axis=0)).float()

    def reduceXYScale(self, xScale=None, yScale=None):
        """Change weights such that it is invariant if we add a scaler ahead and after this network."""
        state_dct = self.state_dict()
        nlyr = len(self.layers)
        if xScale is not None:
            w0 = state_dct['main.0.weight']
            b0 = state_dct['main.0.bias']
            xmean, xstd = xScale
            w0 *= torch.from_numpy(xstd.T).float()
            b0 *= torch.from_numpy(np.squeeze(xstd, axis=0)).float()
            b0 += torch.from_numpy(np.squeeze(ymean, axis=0)).float()
        if yScale is not None:
            wf = state_dct['main.%d.weight' % (nlyr - 1)]
            bf = state_dct['main.%d.bias' % (nlyr - 1)]
            ymean, ystd = yScale
            bf -= torch.mv(wf, torch.from_numpy(np.squeeze(xmean/xstd, axis=0)).float())
            wf /= torch.from_numpy(ystd).float()

    def printWeights(self, lyr=None):
        """Print weights of a layer, or all"""
        if lyr is None:
            val = []
            for tmp in self.parameters():
                print(tmp.data)
                val.append(tmp.data.cpu().numpy())
        else:
            selfparam = list(self.parameters())
            print('layer %d' % lyr, selfparam[lyr].data)
            val = selfparam[lyr].data.cpu().numpy()
        return val

    def __str__(self):
        nm = '_'.join(map(str, self.lyrs))
        return nm


class GaoNetBN(GaoNet):
    """A GaoNet class with batch normalization support."""
    def __init__(self, lyrs):
        nn.Module.__init__(self)
        assert len(lyrs) >= 3
        self.lyrs = lyrs
        self.layers = []
        self.bnlayers = []
        nHidden = len(lyrs) - 2
        lasthidden = lyrs[-2]
        outlayer = lyrs[-1]
        for i in range(nHidden):
            self.layers.append(nn.Linear(lyrs[i], lyrs[i+1]))
            self.bnlayers.append(nn.BatchNorm1d(lyrs[i + 1]))
        # final layer is linear output
        self.layers.append(nn.Linear(lasthidden, outlayer))
        self.layers.extend(self.bnlayers)
        # use the OrderedDict trick to assemble the system
        self.main = nn.Sequential(OrderedDict([(str(i), lyr) for i, lyr in enumerate(self.layers)]))
        self.nHidden = nHidden

    def forward(self, x):
        """Forward pass for the model"""
        out = x
        for i in range(self.nHidden):
            out = F.leaky_relu(self.bnlayers[i](self.layers[i](out)), 0.2)
        # final linear layer
        out = self.layers[self.nHidden](out)
        return out


class autoEncoder(nn.Module):
    """An auto encoder contains both encoder and decoder"""
    def __init__(self, enlyr, delyr, bn=False):
        super(autoEncoder, self).__init__()
        assert enlyr[0] == delyr[-1] and enlyr[-1] == delyr[0]
        assert len(enlyr) > 1 and len(delyr) > 1  # cannot be too simple
        self.enlyr = enlyr
        self.delyr = delyr
        nenlyr = len(enlyr)
        ndelyr = len(delyr)
        self.enlayers = []
        self.delayers = []
        for i in range(nenlyr - 1):
            self.enlayers.append(nn.Linear(enlyr[i], enlyr[i + 1]))
            if i < nenlyr - 1:
                self.enlayers.append(nn.LeakyReLU(0.2))
                self.enlayers.append(nn.BatchNorm1d(enlyr[i + 1]))
        for i in range(ndelyr - 1):
            self.delayers.append(nn.Linear(delyr[i], delyr[i + 1]))
            if i < ndelyr - 1:
                self.delayers.append(nn.LeakyReLU(0.2))
                self.delayers.append(nn.BatchNorm1d(delyr[i + 1]))
        self.encoder = nn.Sequential(OrderedDict([(str(i), lyr) for i, lyr in enumerate(self.enlayers)]))
        self.decoder = nn.Sequential(OrderedDict([(str(i), lyr) for i, lyr in enumerate(self.delayers)]))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def __str__(self):
        nm1 = '_'.join(map(str, self.enlyr))
        nm2 = '_'.join(map(str, self.delyr))
        return '%s-%s' % (nm1, nm2)


def subNetFromNet(net, fraction, depth=None):
    """Given a feed-forward neural network.

    Use the fact that linear scaling of hidden layers introduce both quadratic and linear terms.

    :param net: list, int, specifies the mlp architecture
    :param fraction: float, fraction of data of the subset
    :param depth: int/None, the depth of network (including input and output), if None, equals len(net).
    if the depth != len(net), we have to assume hidden layers are of equal width.
    :return: list, int, the output network architecture
    """
    assert len(net) > 2
    assert 0 <= fraction and fraction <= 1
    d_in = net[0]
    d_out = net[-1]
    # calculate linear term of equation.
    num_lin = d_in * net[1] + d_out * net[-2]
    num_quad = 0 if len(net) == 3 else sum([a * b for a, b in zip(net[1:-2], net[2:-1])])
    if depth is None or depth == len(net) or depth <= 0:  # in this case, we solve fraction of hidden layers
        if num_quad == 0:
            k = fraction
        else:
            sol = np.roots([num_quad, num_lin, -fraction * (num_quad + num_lin)])
            k = float(max(sol[0], sol[1]))
        # generate output
        outnet = [d_in] + [int(k * num) for num in net[1:-1]] + [d_out]
        return outnet
    else:  # in this case, we solve for hidden layer size
        num_param = num_lin + num_quad
        assert depth > 2
        if depth == 3:  # linear case
            n_hidden = num_param * fraction / (d_in + d_out)
        else:
            sol = np.roots([depth - 3, d_in + d_out, -num_param * fraction])
            n_hidden = int(max(sol[0], sol[1]))
        return [d_in] + [int(n_hidden)] * (depth - 2) + [d_out]


def recordStep0(testerror, mdlname, txtname):
    if txtname is None:
        return
    with open(txtname, 'a') as f:
        f.write('model: {}\n initial step {}\n'.format(mdlname, testerror))


def plotError(trainerror, testerror, freq=1, figname=None, merge=False, show=False, txtname=None, mdlname=None, fun=None):
    # make prediction by plot
    if figname is not None:
        if merge:
            fig, ax = plt.subplots()
            axs = [ax, ax]
            colors = ['b', 'r']
        else:
            fig, axs = plt.subplots(2, 1)
            colors = ['b', 'b']
    if fun is not None:
        trainerror = fun(trainerror)
        testerror = fun(testerror)
    if figname is not None:
        axs[0].plot(1+freq*np.arange(len(trainerror)), trainerror, color=colors[0], label='train')
        axs[1].plot(1+freq*np.arange(len(testerror)), testerror, color=colors[1], label='test')
        if not merge:
            axs[0].set_title('mini-batch train error')
            axs[1].set_title('test error')
            plt.tight_layout()
        else:
            axs[0].legend()
        plt.savefig(figname)
        if show:
            plt.show()
    if txtname is not None:
        step = len(trainerror)
        step10 = int(0.9 * step)
        step20 = int(0.8 * step)
        testerror = np.array(testerror)
        trainerror = np.array(trainerror)
        train10, test10 = np.mean(trainerror[step10:step]), np.mean(testerror[step10:step], keepdims=True)
        train20, test20 = np.mean(trainerror[step20:step]), np.mean(testerror[step20:step], keepdims=True)
        trainf = trainerror[-1]
        testf = testerror[-1]
        with open(txtname, 'a') as f:
            f.write('Last one / 10 / 20: train {} / {} / {} test {} / {} / {}\n'.format(trainf, train10, train20, testf, test10, test20))


def model2cpu(model, name='model', ptmode=False, statedict=False):
    """
    Given a model name, convert all necessary stuff into cpu version
    """
    if model.endswith('.pt'):
        ptmode = True
    if ptmode:
        tmp = torch.load(model)
    else:
        with open(model, 'rb') as f:
            tmp = pickle.load(f)
    mdl = tmp[name].cpu()
    xScale = tmp.get('xScale', None)
    yScale = tmp.get('yScale', None)
    if model.endswith('.pt'):
        newpath = model.replace('.pt', '_cpu.pt')
    else:
        newpath = model.replace('.pkl', '_cpu.pt')
    if model.endswith('.pt'):
        newname = model.replace('.pt', '_cpu.pt')
    elif model.endswith('.pkl'):
        newname = model.replace('.pkl', '_cpu.pt')
    else:
        newname = model + '_cpu'
    if statedict:
        torch.save({name: mdl.state_dict(), 'xScale': xScale, 'yScale': yScale}, newname)
    else:
        torch.save({name: mdl, 'xScale': xScale, 'yScale': yScale}, newname)


def modelLoader(model, name='model', reScale=True, cuda=False, ptmode=False, statedict=False, size=None):
    """Given a model name, usually a pickle file, load the model, create a function.
    New argument: reScale, which reads xmean, xstd, umean, ustd from file and properly rescale things
    It returns a function that can be called using !!!RAW!!! data
    """
    if model.endswith('.pt'):
        ptmode = True
    if ptmode:
        tmp = torch.load(model)
    else:
        with open(model, 'rb') as f:
            tmp = pickle.load(f)
    if statedict:
        assert size is not None
        mdl = GaoNet(size)  #TODO: automatically extract the network size
        mdl.load_state_dict(tmp[name])
    else:
        mdl = tmp[name]
    if reScale:
        xScale = tmp.get('xScale', None)
        yScale = tmp.get('yScale', None)
    else:
        xScale, yScale = None, None
    if cuda:
        mdl.cuda()
    else:
        mdl.cpu()
    # enable evaluation mode
    mdl.eval()

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
            feedx = Variable(torch.from_numpy(x.astype(np.float32)), volatile=True)
            if cuda:
                feedx = feedx.cuda()
            predy = mdl(feedx).cpu().data.numpy()
            if yScale is not None:
                predy = destdify(predy, yScale[0], yScale[1])
            if xdim == 1:
                predy = np.squeeze(predy, axis=0)
            return predy
        else:  # we have to do it in batch to save memory or other stuff
            assert batch_size > 0
            nData = len(x)
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
            loopN = nData // batchsize + 1
            if nData % batchsize == 0:
                loopN -= 1
            for i in range(loopN):
                ind0, indf = batchsize*i, batchsize*(i + 1)
                if indf > nData:
                    indf = nData
                tmpy = fun(x[ind0:indf])
                predy[ind0:indf] = tmpy
            # if yScale is not None:
            #     predy = destdify(predy, yScale[0], yScale[1])
            return predy

    return fun


def svcLoader(model, name='model', reScale=True):
    """Load a svm"""
    with open(model, 'rb') as f:
        tmp = pickle.load(f)
    svc = tmp[name]
    xScale = tmp.get('xScale', None)

    # create the function
    def fun(x):
        '''given x, return y, x can be batch or single.'''
        xdim = x.ndim
        if xdim == 1:
            x = np.expand_dims(x, axis=0)  # make it 1 by X
        if xScale is not None:
            x = stdify(x, xScale[0], xScale[1])
        # convert and feed
        predy = svc.predict(x)
        if xdim == 1:
            predy = predy[0]
        return predy

    return fun


def encoderLoader(model, name='model', reScale=True, cuda=False, return_encoder=False, return_decoder=False):
    """Read the autoEncoder and return two functions"""
    if model.endswith('.pt'):
        ptmode = True
    if not ptmode:
        with open(model, 'rb') as f:
            tmp = pickle.load(f)
    else:
        tmp = torch.load(model)
    mdl = tmp[name].cpu()
    if reScale:
        xScale = tmp.get('xScale', None)
    else:
        xScale = None
    if cuda:
        mdl.cuda()
    # create the function

    def encoder(x):
        '''given x, return y, x can be batch or single.'''
        xdim = x.ndim
        if xdim == 1:
            x = np.expand_dims(x, axis=0)  # make it 1 by X
        if xScale is not None:
            x = stdify(x, xScale[0], xScale[1])
        # convert and feed
        feedx = Variable(torch.from_numpy(x)).float()
        if cuda:
            feedx = feedx.cuda()
        predy = mdl.encoder(feedx).cpu().data.numpy()
        if xdim == 1:
            predy = np.squeeze(predy, axis=0)
        return predy

    def decoder(x):
        xdim = x.ndim
        if xdim == 1:
            x = np.expand_dims(x, axis=0)  # make it 1 by X
        # convert and feed
        feedx = Variable(torch.from_numpy(x)).float()
        if cuda:
            feedx = feedx.cuda()
        predy = mdl.decoder(feedx).cpu().data.numpy()
        if xdim == 1:
            predy = np.squeeze(predy, axis=0)
        if xScale is not None:
            predy = destdify(predy, xScale[0], xScale[1])
        return predy

    def recoverer(x):
        return decoder(encoder(x))

    if not (return_encoder or return_decoder):
        return recoverer

    return_val = (recoverer,)
    if return_encoder:
        return_val += (encoder,)
    if  return_decoder:
        return_val += (decoder,)
    return return_val


def modelLoaderV2(model, cudafy=False, ptmode=False):
    """Given a model name, usually a pickle file, load the model. Return a tuple of net and xScale
    It returns the raw Net and corresponding xScale and yScale
    """
    if model.endswith('.pt'):
        ptmode = True
    if not ptmode:
        with open(model, 'rb') as f:
            tmp = pickle.load(f)
    else:
        tmp = torch.load(model)
    mdl, xScale, yScale = tmp.get('model', None), tmp.get('xScale', None), tmp.get('yScale', None)
    assert xScale is not None
    if cudafy:
        if xScale is not None:
            xScale = [Variable(torch.from_numpy(xScale[0].astype(np.float32))).cuda(),
                        Variable(torch.from_numpy(xScale[1].astype(np.float32)).cuda())]
        if yScale is not None:
            yScale = [Variable(torch.from_numpy(yScale[0].astype(np.float32))).cuda(),
                        Variable(torch.from_numpy(yScale[1].astype(np.float32)).cuda())]
        mdl.cuda()
    else:
        mdl.cpu()
        if xScale is not None:
            if isinstance(xScale[0], np.ndarray):
                xScale = [Variable(torch.from_numpy(x.astype(np.float32))) for x in xScale]
        if yScale is not None:
            if isinstance(yScale[0], np.ndarray):
                yScale = [Variable(torch.from_numpy(x.astype(np.float32))) for x in yScale]
    return [mdl, xScale, yScale]


def funGen(net, xScale=None, yScale=None, *args, **kwargs):
    """Give a net, we use scales to get a function"""
    if isinstance(net, list):
        mdl, xScale, yScale = net
    elif isinstance(net, nn.Module):
        mdl = net
    if xScale is None:
        xmean, xstd = 0, 1
    else:
        xmean, xstd = xScale
    if yScale is None:
        ymean, ystd = 0, 1
    else:
        ymean, ystd = yScale

    def fun(x):
        xdim = x.ndim
        if xdim == 1:
            xin = np.expand_dims(x, axis=0)
        else:
            xin = x
        xin = (xin - xmean) / xstd
        feedx = Variable(torch.from_numpy(xin).float(), volatile=True).cuda()
        predy = mdl(feedx, *args, **kwargs).cpu().data.numpy()
        yout = predy * ystd + ymean
        if xdim == 1:
            yout = np.squeeze(yout, axis=0)
        return yout
    return fun


def cplxModelLoader(model, name0='model', name1='vNets', name2='classNet'):
    """
    Given a complex model name, usually a pickle file,
    load all the net models, create a function.
    Based on design of architecture, cplxModel does not have to normalize data
    It returns many functions that can be directly called!!!
    """
    with open(model, 'rb') as f:
        tmp = pickle.load(f)
    mdls = tmp[name0]

    # create the function
    def fun(x, mdltpl):
        '''given x, return y, x can be batch or single.'''
        xdim = x.ndim
        if x.ndim == 1:
            x = np.expand_dims(x, axis=0)  # make it 1 by X
        if mdltpl[1] is not None:
            if not isinstance(mdltpl[1][0], np.ndarray):
                xmean = mdltpl[1][0].cpu().data.numpy()
                xstd = mdltpl[1][1].cpu().data.numpy()
            else:
                xmean, xstd = mdltpl[1][0], mdltpl[1][1]
        # convert and feed
        fdx = (x - xmean) / xstd
        feedx = Variable(torch.from_numpy(fdx.astype(np.float32)))
        predy = mdltpl[0].cpu()(feedx).data.numpy()
        if mdltpl[2] is not None:
            if not isinstance(mdltpl[2][0], np.ndarray):
                ymean = mdltpl[2][0].cpu().data.numpy()
                ystd = mdltpl[2][1].cpu().data.numpy()
            else:
                ymean, ystd = mdltpl[2][0], mdltpl[2][1]
            predy = predy * ystd + ymean
        if xdim == 1:
            predy = np.squeeze(predy, axis=0)
        return predy

    # for vNets
    vNets = mdls[name1]
    classNet = mdls[name2]
    vNetFun = []
    for net in vNets:
        vNetFun.append(partial(fun, mdltpl=net))
    # for classFun
    classFun = partial(fun, mdltpl=classNet)
    return {'vNets': vNetFun, 'classNet': classFun}


def argMaxFunEvaluate(x, classFun, vNetFun):
    """Given a bunch of functions, do argmax prediction."""
    if x.ndim == 1:
        predy = classFun(x)
        lbl = np.argmax(predy)
        return vNetFun[lbl](x)
    elif x.ndim == 2:
        rst = None
        N, dimx = x.shape
        nlbl = len(vNetFun)
        predy = classFun(x)
        lbls = np.argmax(predy, axis=1)
        for lbl in xrange(nlbl):
            ind = np.where(lbls == lbl)[0]
            if len(ind > 0):
                predy = vNetFun[lbl](x[ind])
                if rst is None:
                    dimy = predy.shape[1]
                    rst = np.zeros((N, dimy))
                rst[ind, :] = predy
        return rst


def argMaxMdlEvaluate(x, classNet, vNet, Normalize=True, cuda=False):
    """Given models, evaluate at x"""
    if x.ndim == 1:
        xin = np.expand_dims(x, axis=0)
        if Normalize:
            xin = (xin - classNet[1][0]) / classNet[1][1]
        fdx = Variable(torch.from_numpy(xin.astype(np.float32)))
        if cuda:
            predy = classNet[0].cuda()(fdx.cuda()).cpu().data.numpy()
        else:
            predy = classNet[0].cpu()(fdx).data.numpy()
        # find which mdl I should use
        lbl = np.argmax(predy, axis=1)
        # evaluate that model
        net = vNet[lbl]
        xin = np.expand_dims(x, axis=0)
        if Normalize:
            xin = (xin - net[1][0]) / vNet[1][1]
        fdx = Variable(torch.from_numpy(xin.astype(np.float32)))
        if cuda:
            predy = net[0].cuda()(fdx.cuda()).cpu().data.numpy()
        else:
            predy = net[0].cpu()(fdx).data.numpy()
        # convert back to normal
        if Normalize:
            predy = predy * net[2][1] + net[2][0]
        predy = np.squeeze(predy)
        return predy
    elif x.ndim == 2:
        vy = []
        for x1 in x:
            vy.append(argMaxMdlEvaluate(x, classNet, vNet, Normalize, cuda))
        return np.array(vy)
    else:
        assert Exception("x dim has to be 1 or 2")
