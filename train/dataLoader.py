#! /usr/bin/env python
"""
Here contains my own implementation of dataLoader.
It is based on torch, but finely tuned for my usage.
"""
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import os
import cPickle as pkl


"""
Design philosophy
1. Let a huge class stores raw data
2. Let two small class store data in two sets
3. each small set, we can operate it as others, we iterate, shuffle
"""


def checkstd(args, tol=1e-3):
    '''Find those data with std < 1e-3, we do not change it'''
    if isinstance(args, list):
        for arg in args:
            if isinstance(arg, np.ndarray):
                arg[arg < tol] = 1.0
    elif isinstance(args, np.ndarray):
        args[args < tol] = 1.0
    else:
        raise NotImplementedError


class unaryFactory(Dataset):
    """Factory for only one variable"""
    def __init__(self):
        self.numData = 0
        self._data = None
        raise NotImplementedError

    def shuffle(self, seed=None):
        np.random.seed(seed)
        np.random.shuffle(self._data)

    def __len__(self):
        return self.numData

    def __getitem__(self, idx):
        return self._data[idx]


class Factory(Dataset):
    """For factory"""
    def __init__(self):
        self.numData = 0
        self._xdata = None
        self._ydata = None
        self._xname = None
        self._yname = None
        raise NotImplementedError

    def shuffle(self, seed=None):
        np.random.seed(seed)
        ind = np.arange(self.__len__())
        np.random.shuffle(ind)
        self._xdata, self._ydata = self._xdata[ind], self._ydata[ind]

    def __len__(self):
        return self.numData

    def __getitem__(self, idx):
        sample = {self._xname: self._xdata[idx], self._yname: self._ydata[idx]}
        return sample


class subFactory(Dataset):
    def __init__(self, factory, start, final):
        self.factory = factory
        assert 0 <= start and start <= 1
        assert 0 <= final and final <= 1
        assert start <= final
        self.startind = int(start * len(factory))
        self.finalind = int(final * len(factory))
        if hasattr(factory, 'xmean'):
            self.xmean = factory.xmean
            self.xstd = factory.xstd
        if hasattr(factory, 'ymean'):
            self.ymean = factory.ymean
            self.ystd = factory.ystd
        if hasattr(factory, '_xname'):
            self._xname = factory._xname
        if hasattr(factory, '_yname'):
            self._yname = factory._yname

    def __len__(self):
        return self.finalind - self.startind

    def __getitem__(self, idx):
        return self.factory[idx]

    def allitem(self):
        return self.factory[self.startind:self.finalind]


class unaryKeyFactory(unaryFactory):
    """A factory that contains only one variable"""
    def __init__(self, fnm, xnm, xfun=None, normalize=True):
        if isinstance(fnm, str):
            if '.npz' in fnm or '.npy' in fnm:
                tmp = np.load(fnm)
            elif '.pkl' in fnm:
                tmp = pkl.load(open(fnm, 'rb'))
            else:
                raise NotImplementedError
        else:
            if callable(getattr(fnm, 'keys', None)):
                tmp = fnm
            else:
                if isinstance(fnm, np.ndarray):
                    tmp = fnm
                else:
                    raise NotImplementedError
        if xfun is not None:
            self._data = xfun(tmp[xnm])
        else:
            self._data = tmp[xnm]
        self.numData = len(self._data)
        if normalize:
            xmean, xstd = np.mean(self._data, axis=0, keepdims=True), np.std(self._data, axis=0, keepdims=True)
            checkstd(xstd, tol=1e-3)
            self._realdata = self._data.copy()
        else:
            xmean, xstd = 0, 1
            self._realdata = self._data
        self._data = (self._data - xmean) / xstd
        self._xmean, self._xstd = xmean, xstd
        self._data = self._data.astype(np.float32)  # convert to float
        self.xmean, self.xstd = self._xmean, self._xstd
        self._xname = xnm  # actually useless


class keyFactory(Factory):
    """
    Generate a factory type using key and dict
    """
    def __init__(self, fnm, xnm, ynm, xfun=None, yfun=None, normalize=True):
        # load data, it can be string (np or pkl) or dict
        if isinstance(fnm, str):
            if '.npz' in fnm or '.npy' in fnm:
                tmp = np.load(fnm)
            elif '.pkl' in fnm:
                tmp = pkl.load(open(fnm, 'rb'))
            else:
                raise NotImplementedError
        else:
            if callable(getattr(fnm, 'keys', None)):
                tmp = fnm
            else:
                raise NotImplementedError
        if xfun is not None:
            self._xdata = xfun(tmp[xnm])
        else:
            self._xdata = tmp[xnm]
        if yfun is not None:
            self._ydata = yfun(tmp[ynm])
        else:
            self._ydata = tmp[ynm]
        self.numData = len(self._xdata)
        if normalize:
            xmean, xstd = np.mean(self._xdata, axis=0, keepdims=True), np.std(self._xdata, axis=0, keepdims=True)
            umean, ustd = np.mean(self._ydata, axis=0, keepdims=True), np.std(self._ydata, axis=0, keepdims=True)
            checkstd([xstd, ustd], tol=1e-3)
            self._realxdata = self._xdata.copy()
            self._realydata = self._ydata.copy()
        else:
            xmean, xstd = 0, 1
            umean, ustd = 0, 1
            self._realxdata = self._xdata
            self._realydata = self._ydata
        self._xdata = (self._xdata - xmean) / xstd
        self._ydata = (self._ydata - umean) / ustd
        self._xmean, self._xstd = xmean, xstd
        self._ymean, self._ystd = umean, ustd
        self._xdata = self._xdata.astype(np.float32)  # convert to float
        self._ydata = self._ydata.astype(np.float32)
        self.xmean, self.xstd = self._xmean, self._xstd
        self.ymean, self.ystd = self._ymean, self._ystd
        self._xname = xnm
        self._yname = ynm


class labelFactory(Factory):
    """
    For classification task, we have a bunch of labelled data, use this to manage those.
    We support two modes, one given file, we use a list of keys and functions operated on.
    Another mode is to simply add data label by label.
    """
    def __init__(self, fnm, nameLblPair, xfun=None, normalize=True):
        if isinstance(fnm, str):
            if '.npz' in fnm or '.npy' in fnm:
                tmp = np.load(fnm)
            elif '.pkl' in fnm:
                tmp = pkl.load(open(fnm, 'rb'))
            else:
                raise NotImplementedError
        elif isinstance(fnm, dict):
            tmp = fnm
        self.lstData = []
        self.lstLabel = []
        for nm, lbl in nameLblPair:
            if xfun is None:
                xdata = tmp[nm]
            else:
                xdata = xfun(tmp[nm])
            self.lstData.append(xdata)
            self.lstLabel.append(lbl*np.ones(len(xdata), dtype=np.int))
        # get whole data
        self._realxdata = np.concatenate(self.lstData, axis=0)
        self._label = np.concatenate(self.lstLabel, axis=0)
        self._ydata = self._label
        self.numLabel = len(self.lstLabel)
        if normalize:
            self._xmean = np.mean(self._realxdata, axis=0, keepdims=True)
            self._xstd = np.std(self._realxdata, axis=0, keepdims=True)
            checkstd([self._xstd], tol=1e-3)
            self._xdata = (self._realxdata - self._xmean) / self._xstd
        else:
            self._xmean, self._xstd = 0, 1
            self._xdata = self._realxdata
        self.numData = len(self._xdata)
        self._xdata = self._xdata.astype(np.float32)  # make it float
        self.xmean, self.xstd = self._xmean, self._xstd
        self._xname = 'x'
        self._yname = 'label'


class vecKeyFactory(Dataset):
    """
    Generate a factory type using key and dict
    """
    def __init__(self, fnm, names, funs, norms):
        """
        names, funs, and norms are containers
        funs can be a function if it takes multiple inputs and returns
        """
        if isinstance(fnm, str):
            if 'npz' in fnm or 'npy' in fnm:
                tmp = np.load(fnm)
            elif 'pkl' in fnm:
                tmp = pkl.load(open(fnm, 'rb'))
            else:
                raise NotImplementedError
        elif isinstance(fnm, dict):
            tmp = fnm
        assert isinstance(names, (list, tuple))
        assert isinstance(norms, (list, tuple)) or norms is None
        self._names = names
        self.nData = len(names)
        if isinstance(funs, (list, tuple)):  # we have element-wise jobs to do
            assert len(names) == len(funs)
            self._data = []
            for name, fun in zip(names, funs):
                if fun is None:
                    self._data.append(tmp[name])
                else:
                    self._data.append(fun(tmp[name]))
        elif funs is None:
            self._data = []
            for name in names:
                self._data.append(tmp[name])
        else:  # in this case, we evaluate the function using all data we collected
            self._data = funs(tmp)  # we cannot pass lower-level since we do not know how many we want
        self.numData = len(self._data[0])
        self._data = [dt.astype(np.float32) for dt in self._data]  # convert to float type
        # we finished loading raw data, now we alternatively normalize those
        if norms is None:
            self._mean = [0 for dt in self._data]
            self._std = [1 for dt in self._data]
            self._realdata = [dt for dt in self._data]  # do we really need this?
        else:
            assert len(norms) == len(names)
            self._mean = [np.mean(dt, axis=0, keepdims=True) if norm else 0 for dt, norm in zip(self._data, norms)]
            self._std = [np.std(dt, axis=0, keepdims=True) if norm else 1 for dt, norm in zip(self._data, norms)]
            checkstd(self._std, tol=1e-3)
            self._realdata = [dt.copy() if norm else dt for dt, norm in zip(self._data, norms)]
        # element wise normalize
        self._data = [(self._data[i] - self._mean[i])/self._std[i] for i in range(self.nData)]

    def shuffle(self, seed=None):
        ind = np.arange(self.__len__())
        np.random.seed(seed)
        np.random.shuffle(ind)
        for i, dt in enumerate(self._data):
            self._data[i] = dt[ind]

    def __len__(self):
        return self.numData

    def __getitem__(self, idx):
        sample = {name: dt[idx] for name, dt in zip(self._names, self._data)}
        return sample


class dataLoader(DataLoader):
    """
    Slightly different dataLoader from built-in one by PyTorch
    It supports getNumData and negative batch_size(means all)
    """
    def __init__(self, dtset, batch_size, shuffle):
        if batch_size <= 0:
            batch_size = len(dtset)
        self.dtset = dtset
        if isinstance(dtset, subFactory):
            sampler = SubsetRandomSampler(range(dtset.startind, dtset.finalind))
            DataLoader.__init__(self, dtset, batch_size, shuffle=False, sampler=sampler)
        else:
            DataLoader.__init__(self, dtset, batch_size, shuffle=shuffle)
        if hasattr(dtset, 'xmean'):
            self.xmean = dtset.xmean
            self.xstd = dtset.xstd
        if hasattr(dtset, 'ymean'):
            self.ymean = dtset.ymean
            self.ystd = dtset.ystd
        if hasattr(dtset, '_xname'):
            self._xname = dtset._xname
        if hasattr(dtset, '_yname'):
            self._yname = dtset._yname

    def getNumData(self):
        return self.dtset.__len__()  # number of data, since len is returning # batches
