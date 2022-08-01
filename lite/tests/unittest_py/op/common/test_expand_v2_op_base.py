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
import hypothesis.strategies as st


def sample_program_configs(draw):
    in_shape = draw(st.sampled_from([[1, 1, 1], [2, 1, 4]]))
    Shape = draw(st.sampled_from([[2, 4, 4], [3, 2, 3, 4]]))
    expand_shape = draw(st.sampled_from([[2, 5, 4], [2, 3, 4]]))
    with_Shape = draw(st.sampled_from([True, False]))

    #todo daming5432 input vector tensor
    with_expand_shape = draw(st.sampled_from([False]))

    def generate_shape(*args, **kwargs):
        return np.array(Shape).astype(np.int32)

    def generate_expand_shape(*args, **kwargs):
        return np.array(expand_shape).astype(np.int32)

    def generate_input_int32(*args, **kwargs):
        return np.random.random(in_shape).astype(np.int32)

    def generate_input_int64(*args, **kwargs):
        return np.random.random(in_shape).astype(np.int64)

    def generate_input_float32(*args, **kwargs):
        return np.random.random(in_shape).astype(np.float32)

    input_type = draw(
        st.sampled_from([
            generate_input_int32, generate_input_int64, generate_input_float32
        ]))

    def gnerate_inputs(with_Shape, with_expand_shape):
        inputs1 = {}
        inputs2 = {}
        if (with_Shape and with_expand_shape):
            inputs1 = {
                "X": ["input_data"],
                "Shape": ["shape_data"],
                "expand_shapes_tensor": ["expand_data"]
            }
            inputs2 = {
                "input_data": TensorConfig(data_gen=partial(input_type)),
                "shape_data": TensorConfig(data_gen=partial(generate_shape)),
                "expand_data":
                TensorConfig(data_gen=partial(generate_expand_shape))
            }
        elif ((not with_Shape) and with_expand_shape):
            inputs1 = {
                "X": ["input_data"],
                "expand_shapes_tensor": ["expand_data"]
            }
            inputs2 = {
                "input_data": TensorConfig(data_gen=partial(input_type)),
                "expand_data":
                TensorConfig(data_gen=partial(generate_expand_shape))
            }
        elif (with_Shape and (not with_expand_shape)):
            inputs1 = {"X": ["input_data"], "Shape": ["shape_data"]}
            inputs2 = {
                "input_data": TensorConfig(data_gen=partial(input_type)),
                "shape_data": TensorConfig(data_gen=partial(generate_shape))
            }
        else:
            inputs1 = {"X": ["input_data"]}
            inputs2 = {
                "input_data": TensorConfig(data_gen=partial(input_type))
            }
        return [inputs1, inputs2]

    attr_shape = draw(
        st.sampled_from([[2, 3, 4], [2, 4, 4], [2, 2, 3, 4], [3, 2, 5, 4]]))
    inputs = gnerate_inputs(with_Shape, with_expand_shape)
    expand_v2_op = OpConfig(
        type="expand_v2",
        inputs=inputs[0],
        outputs={"Out": ["output_data"]},
        attrs={"shape": attr_shape})

    program_config = ProgramConfig(
        ops=[expand_v2_op],
        weights={},
        inputs=inputs[1],
        outputs=["output_data"])
    return program_config
