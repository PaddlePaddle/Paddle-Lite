# -*- coding: utf-8 -*-
# This file is part of Eigen, a lightweight C++ template library
# for linear algebra.
#
# Copyright (C) 2009 Benjamin Schindler <bschindler@inf.ethz.ch>
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

# Pretty printers for Eigen::Matrix
# This is still pretty basic as the python extension to gdb is still pretty basic. 
# It cannot handle complex eigen types and it doesn't support many of the other eigen types
# This code supports fixed size as well as dynamic size matrices

# To use it:
#
# * Create a directory and put the file as well as an empty __init__.py in 
#   that directory.
# * Create a ~/.gdbinit file, that contains the following:
#      python
#      import sys
#      sys.path.insert(0, '/path/to/eigen/printer/directory')
#      from printers import register_eigen_printers
#      register_eigen_printers (None)
#      end

import gdb
import re
import itertools
from bisect import bisect_left

# Basic row/column iteration code for use with Sparse and Dense matrices
class _MatrixEntryIterator(object):
	
	def __init__ (self, rows, cols, rowMajor):
		self.rows = rows
		self.cols = cols
		self.currentRow = 0
		self.currentCol = 0
		self.rowMajor = rowMajor

	def __iter__ (self):
		return self

	def next(self):
                return self.__next__()  # Python 2.x compatibility

	def __next__(self):
		row = self.currentRow
		col = self.currentCol
		if self.rowMajor == 0:
			if self.currentCol >= self.cols:
				raise StopIteration
				
			self.currentRow = self.currentRow + 1
			if self.currentRow >= self.rows:
				self.currentRow = 0
				self.currentCol = self.currentCol + 1
		else:
			if self.currentRow >= self.rows:
				raise StopIteration
				
			self.currentCol = self.currentCol + 1
			if self.currentCol >= self.cols:
				self.currentCol = 0
				self.currentRow = self.currentRow + 1

		return (row, col)

class EigenMatrixPrinter:
	"Print Eigen Matrix or Array of some kind"

	def __init__(self, variety, val):
		"Extract all the necessary information"
		
		# Save the variety (presumably "Matrix" or "Array") for later usage
		self.variety = variety
		
		# The gdb extension does not support value template arguments - need to extract them by hand
		type = val.type
		if type.code == gdb.TYPE_CODE_REF:
			type = type.target()
		self.type = type.unqualified().strip_typedefs()
		tag = self.type.tag
		regex = re.compile('\<.*\>')
		m = regex.findall(tag)[0][1:-1]
		template_params = m.split(',')
		template_params = [x.replace(" ", "") for x in template_params]
		
		if template_params[1] == '-0x00000000000000001' or template_params[1] == '-0x000000001' or template_params[1] == '-1':
			self.rows = val['m_storage']['m_rows']
		else:
			self.rows = int(template_params[1])
		
		if template_params[2] == '-0x00000000000000001' or template_params[2] == '-0x000000001' or template_params[2] == '-1':
			self.cols = val['m_storage']['m_cols']
		else:
			self.cols = int(template_params[2])
		
		self.options = 0 # default value
		if len(template_params) > 3:
			self.options = template_params[3];
		
		self.rowMajor = (int(self.options) & 0x1)
		
		self.innerType = self.type.template_argument(0)
		
		self.val = val
		
		# Fixed size matrices have a struct as their storage, so we need to walk through this
		self.data = self.val['m_storage']['m_data']
		if self.data.type.code == gdb.TYPE_CODE_STRUCT:
			self.data = self.data['array']
			self.data = self.data.cast(self.innerType.pointer())
			
	class _iterator(_MatrixEntryIterator):
		def __init__ (self, rows, cols, dataPtr, rowMajor):
			super(EigenMatrixPrinter._iterator, self).__init__(rows, cols, rowMajor)

			self.dataPtr = dataPtr

		def __next__(self):
			
			row, col = super(EigenMatrixPrinter._iterator, self).__next__()
			
			item = self.dataPtr.dereference()
			self.dataPtr = self.dataPtr + 1
			if (self.cols == 1): #if it's a column vector
				return ('[%d]' % (row,), item)
			elif (self.rows == 1): #if it's a row vector
				return ('[%d]' % (col,), item)
			return ('[%d,%d]' % (row, col), item)
			
	def children(self):
		
		return self._iterator(self.rows, self.cols, self.data, self.rowMajor)
		
	def to_string(self):
		return "Eigen::%s<%s,%d,%d,%s> (data ptr: %s)" % (self.variety, self.innerType, self.rows, self.cols, "RowMajor" if self.rowMajor else  "ColMajor", self.data)

class EigenSparseMatrixPrinter:
	"Print an Eigen SparseMatrix"

	def __init__(self, val):
		"Extract all the necessary information"

		type = val.type
		if type.code == gdb.TYPE_CODE_REF:
			type = type.target()
		self.type = type.unqualified().strip_typedefs()
		tag = self.type.tag
		regex = re.compile('\<.*\>')
		m = regex.findall(tag)[0][1:-1]
		template_params = m.split(',')
		template_params = [x.replace(" ", "") for x in template_params]

		self.options = 0
		if len(template_params) > 1:
			self.options = template_params[1];
		
		self.rowMajor = (int(self.options) & 0x1)
		
		self.innerType = self.type.template_argument(0)
		
		self.val = val

		self.data = self.val['m_data']
		self.data = self.data.cast(self.innerType.pointer())

	class _iterator(_MatrixEntryIterator):
		def __init__ (self, rows, cols, val, rowMajor):
			super(EigenSparseMatrixPrinter._iterator, self).__init__(rows, cols, rowMajor)

			self.val = val
			
		def __next__(self):
			
			row, col = super(EigenSparseMatrixPrinter._iterator, self).__next__()
				
			# repeat calculations from SparseMatrix.h:
			outer = row if self.rowMajor else col
			inner = col if self.rowMajor else row
			start = self.val['m_outerIndex'][outer]
			end = ((start + self.val['m_innerNonZeros'][outer]) if self.val['m_innerNonZeros'] else
			       self.val['m_outerIndex'][outer+1])

			# and from CompressedStorage.h:
			data = self.val['m_data']
			if start >= end:
				item = 0
			elif (end > start) and (inner == data['m_indices'][end-1]):
				item = data['m_values'][end-1]
			else:
				# create Python index list from the target range within m_indices
				indices = [data['m_indices'][x] for x in range(int(start), int(end)-1)]
				# find the index with binary search
				idx = int(start) + bisect_left(indices, inner)
				if ((idx < end) and (data['m_indices'][idx] == inner)):
					item = data['m_values'][idx]
				else:
					item = 0

			return ('[%d,%d]' % (row, col), item)

	def children(self):
		if self.data:
			return self._iterator(self.rows(), self.cols(), self.val, self.rowMajor)

		return iter([])   # empty matrix, for now


	def rows(self):
		return self.val['m_outerSize'] if self.rowMajor else self.val['m_innerSize']

	def cols(self):
		return self.val['m_innerSize'] if self.rowMajor else self.val['m_outerSize']

	def to_string(self):

		if self.data:
			status = ("not compressed" if self.val['m_innerNonZeros'] else "compressed")
		else:
			status = "empty"
		dimensions  = "%d x %d" % (self.rows(), self.cols())
		layout      = "row" if self.rowMajor else "column"

		return "Eigen::SparseMatrix<%s>, %s, %s major, %s" % (
			self.innerType, dimensions, layout, status )

class EigenQuaternionPrinter:
	"Print an Eigen Quaternion"
	
	def __init__(self, val):
		"Extract all the necessary information"
		# The gdb extension does not support value template arguments - need to extract them by hand
		type = val.type
		if type.code == gdb.TYPE_CODE_REF:
			type = type.target()
		self.type = type.unqualified().strip_typedefs()
		self.innerType = self.type.template_argument(0)
		self.val = val
		
		# Quaternions have a struct as their storage, so we need to walk through this
		self.data = self.val['m_coeffs']['m_storage']['m_data']['array']
		self.data = self.data.cast(self.innerType.pointer())
			
	class _iterator:
		def __init__ (self, dataPtr):
			self.dataPtr = dataPtr
			self.currentElement = 0
			self.elementNames = ['x', 'y', 'z', 'w']
			
		def __iter__ (self):
			return self
	
		def next(self):
			return self.__next__()  # Python 2.x compatibility

		def __next__(self):
			element = self.currentElement
			
			if self.currentElement >= 4: #there are 4 elements in a quanternion
				raise StopIteration
			
			self.currentElement = self.currentElement + 1
			
			item = self.dataPtr.dereference()
			self.dataPtr = self.dataPtr + 1
			return ('[%s]' % (self.elementNames[element],), item)
			
	def children(self):
		
		return self._iterator(self.data)
	
	def to_string(self):
		return "Eigen::Quaternion<%s> (data ptr: %s)" % (self.innerType, self.data)

def build_eigen_dictionary ():
	pretty_printers_dict[re.compile('^Eigen::Quaternion<.*>$')] = lambda val: EigenQuaternionPrinter(val)
	pretty_printers_dict[re.compile('^Eigen::Matrix<.*>$')] = lambda val: EigenMatrixPrinter("Matrix", val)
	pretty_printers_dict[re.compile('^Eigen::SparseMatrix<.*>$')] = lambda val: EigenSparseMatrixPrinter(val)
	pretty_printers_dict[re.compile('^Eigen::Array<.*>$')]  = lambda val: EigenMatrixPrinter("Array",  val)

def register_eigen_printers(obj):
	"Register eigen pretty-printers with objfile Obj"

	if obj == None:
		obj = gdb
	obj.pretty_printers.append(lookup_function)

def lookup_function(val):
	"Look-up and return a pretty-printer that can print va."
	
	type = val.type
	
	if type.code == gdb.TYPE_CODE_REF:
		type = type.target()
	
	type = type.unqualified().strip_typedefs()
	
	typename = type.tag
	if typename == None:
		return None
	
	for function in pretty_printers_dict:
		if function.search(typename):
			return pretty_printers_dict[function](val)
	
	return None

pretty_printers_dict = {}

build_eigen_dictionary ()
