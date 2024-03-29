#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Gao Tang <gt70@duke.edu>
#
# Distributed under terms of the MIT license.

"""
sympyutil.py

Constains subroutines for better serving of sympy functions.
"""
import sympy as sym
import re
import os
from sympy.utilities.autowrap import autowrap


# extract sparsity pattern from a Jacobian and return three matrices of value, row, col
def getSparseJacobian(jac):
    """Return the sparse jacobian matrix.

    It checks if every term in jac is zero, and return the nonzero terms.
    Caveat: if the term is too complicated, it might cause troubles.

    :param jac: sympy.Matrix, the jacobian matrix we want to find sparsity
    :return val: the sparse vector containing the nnz terms
    :return row: row indexes of the sparse jacobian
    :return col: col indexes of the sparse jacobian
    """
    row, col = jac.shape
    nnz = []
    nnzrow = []
    nnzcol = []
    for i in range(row):
        for j in range(col):
            if jac[i, j] == 0:
                continue
            else:
                nnz.append(jac[i, j])
                nnzrow.append(i)
                nnzcol.append(j)
    print('we found %d nnz' % len(nnz))
    return sym.Matrix(nnz), sym.Matrix(nnzrow), sym.Matrix(nnzcol)


def prettifyCCode(fileName, className=''):
    """Make the C code generated by sympy prettier.

    This functions reads the fileName.h/fileName.c files in working directory and output
    a file named fileName.h.tmp

    :param fileName: str, the file you want to convert, such as XX.h/XX.c
    :param className: str, which class to write to the function names, such as YY::XX()

    """
    funheader = re.compile(r'^[a-zA-Z]+ [a-zA-Z0-9]+\(')
    funbody = re.compile(r'^   [a-zA-Z]+')
    longname = re.compile(r'[a-zA-Z]+(_[0-9]+)')
    appendClassName = className
    outputName = fileName
    headerName = '%s.h' % outputName
    tmpHeaderName = headerName + '.tmp'
    tmpf = open(tmpHeaderName, 'w')
    # process the header file
    with open('%s.h' % outputName) as f:
        for line in f:
            if line[0] in [r'/', r' ', r'#', '\n']:
                continue
            else:
                rst = longname.findall(line)
                if len(rst) > 0:
                    line = line.replace(rst[0], '')
                tmpf.write(line)
    # process the c file
    tmpf.write('\n')
    with open('%s.c' % outputName) as f:
        for i in range(7):
            f.readline()
        while True:
            rst = f.readline()
            if len(rst) == 0:
                break
            elif rst[0] in ['\n', '#']:
                continue
            match = longname.findall(rst)
            if len(match) > 0:
                rst = rst.replace(match[0], '')
            if funheader.match(rst):
                if len(appendClassName) > 0:
                    line = rst.replace(' ', ' %s::' % appendClassName, 1)
                    tmpf.write(line)
                else:
                    tmpf.write(rst)
            elif funbody.match(rst):
                newline = rst.replace(' ', '  ', 1)
                tmpf.write(newline)
            else:
                tmpf.write(rst)
            if rst[0] == '}':
                tmpf.write('\n')


def dumpExpression(savelist, fixvar, moduleName='tmp'):
    """Dump several expressions onto disk and ready to load them all.

    The user inputs a list of sympy expressions and a list of symbols that are considered fixed,
    For each expression, it extracts free symbols and category those into fixed and free, an API
    is generated such that free variables are used as class API, fixed are automatically appended.
    Essentially it generates a folder containing the compiled functions and a wrapper file.

    :param savelist: list of tuple (name, exp), list of expressions to be dumped
    :fixvar: list of global variables. They will not be in function arguments
    :moduleName: str, name of the module. Files are stored in the folder.

    """
    # use re to find the number
    numFder = re.compile('wrapper_module_([0-9]+)')
    usepath = os.path.abspath('./%s' % moduleName)
    if not os.path.exists(usepath):
        os.mkdir(usepath)
    # step 1, get all symbols
    fixSet = set(fixvar)
    # step 2, analysis each expression
    lstFreeSymbols = []  # store freesymbols
    lstFixSymbols = []  # store fixsymbols
    lstFlatten = []
    for i, (name, exp) in enumerate(savelist):
        print('expression %d named %s' % (i, name))
        try:
            print('shape {}'.format(exp.shape))
        except:
            pass
        expsymbols = exp.free_symbols
        usefreesymbols = list(expsymbols.difference(fixSet))
        usefreesymbols.sort(key=str)
        lstFreeSymbols.append(usefreesymbols)
        # find symbols that are not in fixSet
        usefixsymbols = list(expsymbols.intersection(fixSet))
        usefixsymbols.sort(key=str)
        lstFixSymbols.append(usefixsymbols)
        expsymbols = usefreesymbols + usefixsymbols
        tmp = autowrap(exp, language='C', backend='cython', args=expsymbols, tempdir=usepath)
        lstFlatten.append(False)
        if isinstance(exp, sym.Matrix):
            if exp.shape[1] == 1:
                lstFlatten[-1] = True
    # step 3, generate a wrapper for the module we desire
    filename = '%s_wrapper_pre.py' % moduleName
    f = open(filename, 'w')
    f.write('#! /usr/bin/env python\n')
    # import those symbols as understandable names
    spaces = r'    '
    nExp = len(savelist)
    for i in range(nExp):
        f.write('from %s.wrapper_module_%d import autofunc_c as %s\n' % (moduleName, i, savelist[i][0]))
    f.write('\n\nclass %s(object):\n' % moduleName)
    f.write('%s"""A wrapper class that loads sympy symbols."""\n' % spaces)
    f.write('%sdef __init__(self):\n' % spaces)
    # initialize all fixed symbols
    for fixsym in fixSet:
        f.write('%s%sself.%s = 0\n' % (spaces, spaces, fixsym))
    # write functions for those expressions
    for (name, exp), freesym, fixsym, flat in zip(savelist, lstFreeSymbols, lstFixSymbols, lstFlatten):
        arguments = ', '.join([str(symb) for symb in freesym])
        fixarguments = ', '.join(['self.%s' % symb for symb in fixsym])
        if len(arguments) > 0:
            f.write('\n%sdef _%s(self, %s):\n' % (spaces, name, arguments))
        else:
            f.write('\n%sdef _%s(self):\n' % (spaces, name))
        if len(arguments) > 0:
            if len(fixarguments) > 0:
                allarguments = arguments + ', ' + fixarguments
            else:
                allarguments = fixarguments
        else:
            allarguments = fixarguments
        if flat:
            f.write('%s%sreturn %s(%s).flatten()\n' % (spaces, spaces, name, allarguments))
        else:
            f.write('%s%sreturn %s(%s)\n' % (spaces, spaces, name, allarguments))
