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

from program_config import TensorConfig, ProgramConfig, OpConfig, CxxConfig, TargetType, PrecisionType, DataLayoutType, Place
import numpy as np
from functools import partial
from typing import Optional, List, Callable, Dict, Any, Set
import unittest

import hypothesis
from hypothesis import given, settings, seed, example, assume, reproduce_failure
import hypothesis.strategies as st


def check_broadcast(x_shape, y_shape, axis):
    if x_shape == y_shape:
        return True
    else:
        x_dims_size = len(x_shape)
        y_dims_size = len(y_shape)
        max_dims_size = max(x_dims_size, y_dims_size)
        if (x_dims_size == y_dims_size):
            # The axis should be -1 or 0 while the dimension of tensor X is equal to the dimension of tensor Y.
            if not (axis == -1 or axis == 0):
                return False
        # The axis should be met [-max(x_dims_size, y_dims_size), max(x_dims_size, y_dims_size))
        if not (axis >= (-max_dims_size) and axis < max_dims_size):
            return False
        if axis < 0:
            axis = abs(x_dims_size - y_dims_size) + axis + 1
        if not (axis >= 0 and axis < max_dims_size):
            return False
        # The axis should be met axis + min(x_dims_size, y_dims_size) <= max(x_dims_size, y_dims_size)
        if axis + min(x_dims_size, y_dims_size) > max_dims_size:
            return False
        x_dims_array = [1 for i in range(max_dims_size)]
        y_dims_array = [1 for i in range(max_dims_size)]
        if x_dims_size > y_dims_size:
            x_dims_array = x_shape
            for i in range(y_dims_size):
                y_dims_array[axis + i] = y_shape[i]
        else:
            y_dims_array = y_shape
            for i in range(x_dims_size):
                x_dims_array[axis + i] = x_shape[i]
        for i in range(max_dims_size):
            broadcast_flag = (x_dims_array[i] == y_dims_array[i]) or (
                x_dims_array[i] <= 1) or (y_dims_array[i] <= 1)
            if not broadcast_flag:
                return False
        return True
