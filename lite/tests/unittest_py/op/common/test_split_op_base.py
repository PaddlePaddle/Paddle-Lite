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
from hypothesis import given, settings, seed, example, assume
import hypothesis.strategies as st


def sample_program_configs(draw):
    in_shape = draw(
        st.sampled_from([[3, 3, 24], [3, 24, 24], [3, 24], [24, 24], [24]]))
    batch = draw(st.integers(min_value=1, max_value=10))
    in_shape.insert(0, batch)
    sections = draw(
        st.sampled_from([[], [1, 2], [2, 1], [10, 14], [1, 1, 1], [2, 2, 2],
                         [3, 3, 3], [3, 7, 14]]))
    input_num = draw(st.sampled_from([0, 1]))
    num = draw(st.sampled_from([0, 3]))
    input_axis = draw(st.sampled_from([0, 1, 2, 3]))
    input_type = draw(
        st.sampled_from(["type_float", "type_int", "type_int64"]))
    Out = draw(
        st.sampled_from([["output_var0", "output_var1"],
                         ["output_var0", "output_var1", "output_var2"]]))

    #Sections and num cannot both be equal to 0.
    assume((num != 0 & len(sections) == 0) | (num == 0 & len(sections) != 0))

    # the dimensions of input and axis match
    assume(input_axis < len(in_shape))

    #When sections and num are not both equal to 0, sections has higher priority.
    #The sum of sections should be equal to the input size.
    if len(sections) != 0:
        assume(len(Out) == len(sections))
        assume(in_shape[input_axis] % len(sections) == 0)
        sum = 0
        for num in sections:
            sum += num
        assume(sum == in_shape[input_axis])

    if num != 0:
        assume(len(Out) == num)
        assume(in_shape[input_axis] % 3 == 0)

    if input_num == 0:
        assume((len(in_shape) == 2) & (in_shape[1] == 24) & (
            sections == [10, 14]) & (len(Out) == 2))

    def generate_input(*args, **kwargs):
        if input_type == "type_float":
            return np.random.normal(0.0, 1.0, in_shape).astype(np.float32)
        elif input_type == "type_int":
            return np.random.normal(0.0, 1.0, in_shape).astype(np.int32)
        elif input_type == "type_int64":
            return np.random.normal(0.0, 1.0, in_shape).astype(np.int64)

    def generate_AxisTensor(*args, **kwargs):
        return np.ones([1]).astype(np.int32)

    def generate_SectionsTensorList1(*args, **kwargs):
        return np.array([10]).astype(np.int32)

    def generate_SectionsTensorList2(*args, **kwargs):
        return np.array([14]).astype(np.int32)

    dics_intput = [{
        "X": ["input_data"],
        "AxisTensor": ["AxisTensor"],
        "SectionsTensorList": ["SectionsTensorList1", "SectionsTensorList2"]
    }, {
        "X": ["input_data"]
    }]

    dics_weight = [{
        "AxisTensor": TensorConfig(data_gen=partial(generate_AxisTensor)),
        "SectionsTensorList1":
        TensorConfig(data_gen=partial(generate_SectionsTensorList1)),
        "SectionsTensorList2":
        TensorConfig(data_gen=partial(generate_SectionsTensorList2))
    }, {}]

    ops_config = OpConfig(
        type="split",
        inputs=dics_intput[input_num],
        outputs={"Out": Out},
        attrs={"sections": sections,
               "num": num,
               "axis": input_axis})

    program_config = ProgramConfig(
        ops=[ops_config],
        weights=dics_weight[input_num],
        inputs={"input_data": TensorConfig(data_gen=partial(generate_input))},
        outputs=Out)

    return program_config
