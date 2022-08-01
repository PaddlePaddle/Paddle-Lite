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


def UpdatePaddingAndDilation(in_shape, weight_shape, paddings, dilations,
                             groups, padding_algorithm, strides):
    paddings4 = []
    if len(paddings) == 2:
        paddings4 = [paddings[0], paddings[0], paddings[1], paddings[1]]
    else:
        paddings4 = paddings

    if padding_algorithm == "SAME":
        for i in range(0, 2):
            out_size = (in_shape[i + 2] + strides[i] - 1) / strides[i]
            pad_sum = max((out_size - 1) * strides[i] + weight_shape[i + 2] -
                          in_shape[i + 2], 0)
            pad_0 = pad_sum / 2
            pad_1 = pad_sum - pad_0
            paddings4[i * 2] = pad_0
            paddings4[i * 2 + 1] = pad_1
            dilations[i] = 1
    elif padding_algorithm == "VALID":
        return [0, 0, 0, 0], dilations
    return paddings4, dilations


def ConvOutputSize(in_shape, weight_shape, dilations, paddings, strides):
    out = []
    for i in range(0, 2):
        dkernel = dilations[i] * (weight_shape[i + 2] - 1) + 1
        out.append((in_shape[i + 2] + paddings[i * 2] + paddings[i * 2 + 1] -
                    dkernel) / strides[i] + 1)
    return int(out[0]), int(out[1])


def ConvTransposeOutputSize(in_shape, weight_shape, dilations, paddings,
                            strides):
    oh = (in_shape[2] - 1) * strides[0] - paddings[0] - paddings[1] + (
        dilations[0] * (weight_shape[2] - 1) + 1)
    ow = (in_shape[3] - 1) * strides[1] - paddings[2] - paddings[3] + (
        dilations[1] * (weight_shape[3] - 1) + 1)
    return oh, ow
