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
    in_shape = draw(
        st.lists(
            st.integers(
                min_value=1, max_value=8), min_size=1, max_size=4))

    bias = draw(st.floats(min_value=-5, max_value=5))
    bias_after_scale = draw(st.booleans())
    scale = draw(st.floats(min_value=-5, max_value=5))
    input_type = draw(st.sampled_from(["int8", "int32", "int64", "float32"]))
    has_scale_tensor = draw(st.booleans())

    def generate_input_int8(*args, **kwargs):
        return np.random.randint(1, 10, in_shape).astype(np.int8)

    def generate_input_int32(*args, **kwargs):
        return np.random.randint(-10, 10, in_shape).astype(np.int32)

    def generate_input_int64(*args, **kwargs):
        return np.random.randint(-10, 10, in_shape).astype(np.int64)

    def generate_input_float32(*args, **kwargs):
        return np.random.random(in_shape).astype(np.float32)

    input_dict = {"X": ["input_data"]}
    if has_scale_tensor:
        input_dict["ScaleTensor"] = "scale_tensor_data"

    scale_op = OpConfig(
        type="scale",
        inputs=input_dict,
        outputs={"Out": ["output_data"]},
        attrs={
            "bias": bias,
            "bias_after_scale": bias_after_scale,
            "scale": scale
        })

    if input_type == "int8":
        if has_scale_tensor:
            program_config = ProgramConfig(
                ops=[scale_op],
                weights={},
                inputs={
                    "input_data":
                    TensorConfig(data_gen=partial(generate_input_int8)),
                    "scale_tensor_data": TensorConfig(shape=[1, ])
                },
                outputs=["output_data"])
        else:
            program_config = ProgramConfig(
                ops=[scale_op],
                weights={},
                inputs={
                    "input_data":
                    TensorConfig(data_gen=partial(generate_input_int8))
                },
                outputs=["output_data"])
    elif input_type == "int32":
        if has_scale_tensor:
            program_config = ProgramConfig(
                ops=[scale_op],
                weights={},
                inputs={
                    "input_data":
                    TensorConfig(data_gen=partial(generate_input_int32)),
                    "scale_tensor_data": TensorConfig(shape=[1, ])
                },
                outputs=["output_data"])
        else:
            program_config = ProgramConfig(
                ops=[scale_op],
                weights={},
                inputs={
                    "input_data":
                    TensorConfig(data_gen=partial(generate_input_int32))
                },
                outputs=["output_data"])
    elif input_type == "int64":
        if has_scale_tensor:
            program_config = ProgramConfig(
                ops=[scale_op],
                weights={},
                inputs={
                    "input_data":
                    TensorConfig(data_gen=partial(generate_input_int64)),
                    "scale_tensor_data": TensorConfig(shape=[1, ])
                },
                outputs=["output_data"])
        else:
            program_config = ProgramConfig(
                ops=[scale_op],
                weights={},
                inputs={
                    "input_data":
                    TensorConfig(data_gen=partial(generate_input_int64))
                },
                outputs=["output_data"])
    elif input_type == "float32":
        if has_scale_tensor:
            program_config = ProgramConfig(
                ops=[scale_op],
                weights={},
                inputs={
                    "input_data":
                    TensorConfig(data_gen=partial(generate_input_float32)),
                    "scale_tensor_data": TensorConfig(shape=[1, ])
                },
                outputs=["output_data"])
        else:
            program_config = ProgramConfig(
                ops=[scale_op],
                weights={},
                inputs={
                    "input_data":
                    TensorConfig(data_gen=partial(generate_input_float32))
                },
                outputs=["output_data"])

    return program_config
