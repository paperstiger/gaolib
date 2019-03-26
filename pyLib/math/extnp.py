#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Gao Tang <gt70@duke.edu>
#
# Distributed under terms of the MIT license.
import numpy as np


def blockIndex(i, j, rows, cols, order='C'):
	"""For a matrix block, we return the index of row and columns.
	
	For a matrix we choose a block using the upper left corner positioned 
	at (i, j) and size (row, col). Each element of the block has row and 
	col index, they are returned in two arrays. The order controls we use
	row or column major order

	For example, blockIndex(1, 3, 2, 3, 'C') returns 
	(array([1, 1, 1, 2, 2, 2]), array([3, 4, 5, 3, 4, 5]))
	:param i: int, the row of the upper left corner
	:param j: int, the column of the upper left corner
	:param rows: int, number of rows of the block
	:param cols: int, number of columns of the block
	:param order, char, ('C'/'F') if we return row or column major
	"""
	if order == 'C':
		row = i + (np.arange(rows)[:, np.newaxis] + np.zeros(cols)).flatten()
		col = j + (np.zeros(rows)[:, np.newaxis] + np.arange(cols)).flatten()
	elif order == 'F':
		row = i + (np.zeros(cols)[:, np.newaxis] + np.arange(rows)).flatten()
		col = j + (np.arange(cols)[:, np.newaxis] + np.zeros(rows)).flatten()
	else:
		raise Exception("Unsupported order")
	return row, col


if __name__ == '__main__':
	print(blockIndex(1, 2, 2, 3, 'C'))
	print(blockIndex(1, 2, 2, 3, 'F'))
