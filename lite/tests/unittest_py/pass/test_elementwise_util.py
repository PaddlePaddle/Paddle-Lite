# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import sys
sys.path.append('..')

from auto_scan_test import FusePassAutoScanTest, IgnoreReasons
from program_config import TensorConfig, ProgramConfig, OpConfig, CxxConfig, TargetType, PrecisionType, DataLayoutType, Place
import numpy as np
from functools import partial
from typing import Optional, List, Callable, Dict, Any, Set
import unittest

import hypothesis
from hypothesis import given, settings, seed, example, assume, reproduce_failure
import hypothesis.strategies as st


def trim_trailing_singular_dims(dims):
    actual_dims_size = len(dims)
    i = actual_dims_size
    for i in range(actual_dims_size, 0, -1):
        if dims[i - 1] != 1:
            break
    if i == len(dims):
        return dims
    trim_dims = []
    for j in range(0, i):
        trim_dims.append(dims[i])
    return trim_dims


def check_input_shape_available(in_shape_x=[],
                                in_shape_y=[],
                                axis=[],
                                out_shape=[]):
    #infer shape
    max_dim = max(len(in_shape_x), len(in_shape_y))
    if len(in_shape_x) == len(in_shape_y):
        if not axis == -1 or axis == 0:
            return False
    if not axis >= -max_dim and axis < max_dim:
        return False
    axis_ = abs(len(in_shape_x) - len(
        in_shape_y)) + axis + 1 if axis < 0 else axis
    #GetBroadcastDimsArrays
    if not axis_ >= 0:
        return False
    if not axis_ < max_dim:
        return False
    x_dims_array = []
    y_dims_array = []
    if (len(in_shape_x) > len(in_shape_y)):
        x_dims_array = in_shape_x
        for i in range(0, axis_):
            y_dims_array.append(1)
        y_dims_array = y_dims_array + in_shape_y
        if not axis_ + len(in_shape_y) < max_dim:  #Paddle error???
            return False
        if axis_ + len(in_shape_y) < max_dim:
            for i in range(axis_ + len(in_shape_y), max_dim):
                y_dims_array.append(1)
    else:
        y_dims_array = in_shape_y
        for i in range(0, axis_):
            x_dims_array.append(1)
        x_dims_array = x_dims_array + in_shape_x
        if not axis_ + len(in_shape_x) < max_dim:  #Paddle error???
            return False
        if axis_ + len(in_shape_x) < max_dim:
            for i in range(axis_ + len(in_shape_x), max_dim):
                x_dims_array.append(1)
    for i in range(0, max_dim):
        if not (x_dims_array[i] == y_dims_array[i] or x_dims_array[i] <= 1 or
                y_dims_array[i] <= 1):
            return False
        out_shape.append(max(x_dims_array[i], y_dims_array[i]))
    #ElementwiseComputeEx
    axis_ = abs(len(in_shape_x) - len(in_shape_y)) if axis == -1 else axis
    if not axis_ >= 0:
        return False
    if not axis_ < max_dim:
        return False
    if len(in_shape_x) > len(in_shape_y):
        y_dims_trimed = trim_trailing_singular_dims(in_shape_y)
        axis_trim = in_shape_x if len(y_dims_trimed) == 0 else axis_
        for i in range(len(y_dims_trimed)):
            if not i + axis_trim < len(in_shape_x):  # Paddle error???
                return False
            if in_shape_x[i + axis_trim] != y_dims_trimed[i]:
                if not (in_shape_x[i + axis_trim] == 1 or
                        y_dims_trimed[i] == 1):
                    return False
    else:
        x_dims_trimed = trim_trailing_singular_dims(in_shape_x)
        axis_trim = in_shape_y if len(x_dims_trimed) == 0 else axis_
        for i in range(len(x_dims_trimed)):
            if not (i + axis_trim < len(in_shape_y)):  # Paddle error???
                return False
            if in_shape_y[i + axis_trim] != x_dims_trimed[i]:
                if not (in_shape_y[i + axis_trim] == 1 or
                        x_dims_trimed[i] == 1):
                    return False

    return True
