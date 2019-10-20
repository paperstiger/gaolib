#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Gao Tang <gt70@duke.edu>
#
# Distributed under terms of the MIT license.

"""
codeio.py

I/O in terms of code such as load a module from another directory
"""
from __future__ import print_function, division
import sys, os, time


def load_source(module_name, file_path):
    """Load a module from a source file with selected name.

    :param module_name: str, the name of the module
    :param file_path: str, the path to the source file
    """
    if not os.path.exists(file_path):
        print('File %s does not exist' % file_path)
        return None
    if sys.version_info[0] == 2:
        import imp
        module = imp.load_source(module_name, file_path)
        return module
    else:
        import importlib.machinery
        import importlib.util
        loader = importlib.machinery.SourceFileLoader(module_name, file_path)
        spec = importlib.machinery.ModuleSpec(module_name, loader, origin=file_path)
        # Basically what import does when there is no loader.create_module().
        module = importlib.util.module_from_spec(spec)
        loader.exec_module(module)
        return module
