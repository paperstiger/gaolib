#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Gao Tang <gt70@duke.edu>
#
# Distributed under terms of the MIT license.

"""
sharedmemory.py

A module that supports shared memory.
"""
import numpy as np
import SharedArray as sa
import re
import warnings

from .fileio import ddctParse


class SharedMemory(object):
    """A class for managing shared memory.

    We use a folder style format for managing files, folder names means a dict.
    The user is able to add ndarray and dict as file, folder
    The user is able to read any file, folder, too. It returns ndarray and dict

    """
    def __init__(self):
        self.keys = []  # stores all keys to arrays
        # update on Aug 22, previous API is too redious and does not support query using only major name
        # my update will be keep a dictionary that maintains name/dict
        self.dct_keys = {}  # use a hierarchical way to manage names
        self.loadMemory()

    def loadMemory(self):
        """Call the sa api to get all names."""
        self.keys = [tmp.name for tmp in sa.list()]
        if len(self.keys) > 0 and isinstance(self.keys[0], bytes):
            self.keys = [key.decode('utf-8') for key in self.keys]
        # also re_construct dct_keys
        slash_finder = re.compile('^(\w+\\\\)')

        def assign_key_to_dct(dct, key, header):
            """For some keys, convert to corresponding things such as keys of dict.

            :param dct: a dictionary to write data into
            :param key: a key being processed
            :param header:  str, the header, header\\\\ is removed
            """
            if header is not None:
                use_key = key.replace(header, '')  # remove those slash
            else:
                use_key = key
            grp = slash_finder.findall(use_key)
            if len(grp) == 0:
                dct[use_key] = None
            else:
                if header is None:
                    header = grp[0].replace('\\', '')
                else:
                    header = header.replace('\\', '')
                if not header in dct:
                    dct[header] = {}
                assign_key_to_dct(dct[header], use_key, grp[0])

        for key in self.keys:
            assign_key_to_dct(self.dct_keys, key, None)

    def getObject(self, name):
        """Retrieve an object by name.

        We first check if it is a name to the file, by checking if it is in self.keys
        Then we check if we refer to a folder by checking all names begins with name/
        After getting a list of keys, we construct arrays / dicts

        """
        return self._getObject(name, self.keys)

    def reportKeys(self):
        """Hierarchically report keys"""
        # print(self.keys)
        def get_key(dct):
            return [(key, get_key(dct[key])) if isinstance(dct[key], dict) else key for key in dct.keys()]
        print(self.keys)
        print(self.dct_keys.keys())
        print(get_key(self.dct_keys))

    def addObject(self, obj, name=None, noalter=False, replace=False, key_dst=None, shortname=None):
        """Add an object this obj can be either ndarray or dict of ndarray.

        obj: ndarray/dict, the object to store
        name: str, the name to store, if it is in type of xx\\yy, I only keep yy
        noalter: bool, if a key exists, we do nothing
        replace: bool, if true and a key exists, we erase previous one and rewrite
        key_dst: list/None, the destination to write a key to
        shortname: str, a shortname, for the new API
        """
        assert shortname is not None
        assert name is not None
        if isinstance(obj, np.ndarray):
            if name in self.keys:
                if noalter:
                    return
                else:
                    if not replace:
                        warnings.warn('name %s exists and not replaced' % name)
                    else:
                        warnings.warn('name %s exists and is erased' % name)
                        self.clear(name)
            register_key = self._getUnregisteredKey(name)
            self.keys.append(register_key)
            if key_dst is None:
                self.dct_keys[shortname] = None  # for an ndarray, the dict maps to None
            else:
                key_dst[shortname] = None
            tmp = sa.create(register_key, obj.shape, dtype=obj.dtype)
            tmp[:] = obj
        elif isinstance(obj, dict):
            to_append = {}
            if key_dst is None:  # we are at root
                self.dct_keys[shortname] = to_append  # for a dict, we add a list  # TODO: add dict of dict type data
            else:
                key_dst[shortname] = to_append
            for key in obj.keys():
                register_key = self._getConcatKey(key, folder=name)
                self.addObject(obj[key], register_key, noalter, replace, to_append, key)  # Here we use key as shortname

    def _getObject(self, name, namespace):
        """Get object by name."""
        flter = [x for x in self.keys if x.startswith(name)]
        if len(flter) == 0:
            return None
        elif len(flter) == 1 and len(name) == len(flter[0]):
            return sa.attach(name)
        else:  # create a new namespace, search for new names
            rst = dict()
            flter = [x for x in flter if x.startswith(name + '\\')]
            lencurname = len(name) + 1
            for name_ in flter:
                spl_names = name_[lencurname:].split('\\')
                rst[spl_names[0]] = self._getObject('%s\\%s' % (name, spl_names[0]), flter)
            if len(rst) == 0:
                return None
            return rst

    def _getConcatKey(self, key, folder):
        """Concatenate folder and key"""
        if not folder.endswith('\\'):
            key = '%s\\%s' % (folder, key)
        else:
            key = folder + key
        return key

    def _getUnregisteredKey(self, key, folder=None):
        """Return an unused key.

        :param key: a tentative name
        :param folder: the folder name to be appended

        """
        if folder is not None:
            key = self._getConcatKey(key, folder)
        register_key = key
        post_val = 1
        while True:
            if register_key not in self.keys:
                break
            else:
                register_key = '%s(%d)' % (key, post_val)
                post_val += 1
        return register_key

    def clear(self, name=None):  # I previously wrote a __del__ function but it will automatically clear memory after script call
        if name is None:
            for key in self.keys:
                sa.delete(key)
            self.keys = []
        else:
            delkeys = [x for x in self.keys if x.startswith(name)]
            for key in delkeys:
                sa.delete(key)
            self.keys = [x for x in self.keys if not x.startswith(name)]

    def clean(self):
        """Clean those attributes with (x)"""
        pattern = re.compile(r'\([0-9]+\)')
        for key in self.keys:
            if len(pattern.findall(key)) > 0:
                print('clean %s' % key)
                self.clear(key)


# create a guy that is globally used
__sm__ = SharedMemory()


# create useful functions
def add(obj, name=None, noalter=False, replace=True):
    """Add an object to the environment. If name is None, it is directly added to root.

    Parameters
    ----------
    obj : ndarray / dict / dict of dict, the data waiting to be added
    name : str, name assigned to the object, if it is None and obj is dict, its keys are added to root
    noalter : bool, if set to True, existing keys will not be modified
    replace : bool, only when noalter is False, if set to True, it replace a data instead of creating a one with non-conflict name

    """
    __sm__.addObject(obj, name, noalter, replace, None, name)


def get(name):
    """Get an object from the environment"""
    return __sm__.getObject(name)

def has(name):
    """Check if a name is in keys"""
    return name in __sm__.keys or name in __sm__.dct_keys.keys()


def clear():
    """This clear current environment."""
    __sm__.clear()


def clean():
    """Clean redundant entries"""
    __sm__.clean()


def list():
    """Show all keys."""
    __sm__.reportKeys()


def clearSA():
    for key in sa.list():
        sa.delete(key.name)


def npload(fnm, sm_name, re_load=False):
    """Load a numpy array / dct of numpy array and keep it in memory, with option of rewriting

    :param fnm: str, file name to the array
    :param sm_name: the name used for saving this array
    """
    if not has(sm_name) or re_load:
        data = ddctParse(fnm)
        add(data, sm_name)  # since we load, we always replace
        return data
    else:
        return get(sm_name)
