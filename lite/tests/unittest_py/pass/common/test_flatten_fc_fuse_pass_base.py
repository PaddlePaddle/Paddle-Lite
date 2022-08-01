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


def sample_program_configs(draw):
    in_shape = draw(
        st.lists(
            st.integers(
                min_value=1, max_value=8), min_size=4, max_size=4))
    in_num_col_dims = draw(st.integers(min_value=1, max_value=1))
    padding_weights = draw(st.integers(min_value=1, max_value=1))
    start_axis = draw(st.integers(min_value=0, max_value=len(in_shape) - 1))
    stop_axis = draw(
        st.integers(
            min_value=start_axis, max_value=len(in_shape) - 1))
    assume((stop_axis - start_axis) == 2)
    start_axis = 1
    if start_axis == 0:
        flatten_out_shape = [
            in_shape[0] * in_shape[1] * in_shape[2], in_shape[3]
        ]
    else:
        flatten_out_shape = [
            in_shape[0], in_shape[1] * in_shape[2] * in_shape[3]
        ]

    weights_0 = 1
    weights_1 = 1
    for i in range(len(flatten_out_shape)):
        if (i < in_num_col_dims):
            weights_1 = weights_1 * flatten_out_shape[i]
        else:
            weights_0 = weights_0 * flatten_out_shape[i]
    weights_shape = [weights_0, weights_1]
    bias_shape = [weights_1]

    flatten_op = OpConfig(
        type='flatten_contiguous_range',
        inputs={"X": ["input_data_x"]},
        outputs={"Out": ["flatten_output_data"],
                 "XShape": ["xshape_data"]},
        attrs={
            "data_format": 'nchw',
            "start_axis": start_axis,
            "stop_axis": stop_axis
        })

    fc_inputs = {}
    program_inputs = {}

    def generate_weights(*args, **kwargs):
        return (np.random.random(weights_shape).astype(np.float32) - 0.5) * 2

    def generate_bias(*args, **kwargs):
        return (np.random.random(bias_shape).astype(np.float32) - 0.5) * 2

    with_bias = draw(st.sampled_from([True]))  #pass require with_bias as True

    act_type = ""
    if (with_bias and np.random.random() > 0.5):
        act_type = "relu"
    if (with_bias):
        fc_inputs = {
            "Input": ["flatten_output_data"],
            "W": ["weights_data"],
            "Bias": ["bias_data"]
        }
        program_inputs = {
            "input_data_x": TensorConfig(shape=in_shape),
            "weights_data": TensorConfig(data_gen=partial(generate_weights)),
            "bias_data": TensorConfig(data_gen=partial(generate_bias))
        }
    else:
        fc_inputs = {"Input": ["flatten_output_data"], "W": ["weights_data"]}
        program_inputs = {
            "input_data_x": TensorConfig(shape=in_shape),
            "weights_data": TensorConfig(data_gen=partial(generate_weights))
        }

    fc_op = OpConfig(
        type='fc',
        inputs=fc_inputs,
        outputs={"Out": ["output_data"]},
        attrs={
            "in_num_col_dims": in_num_col_dims,
            "padding_weights": padding_weights,
            "activation_type": act_type,
            "use_mkldnn": False,
            "padding_weights": False,
            "use_quantizer": False,
            "Scale_in": float(1),
            "Scale_weights": [float(1)],
            "Scale_out": float(1)
        })

    ops = [flatten_op, fc_op]
    program_config = ProgramConfig(
        ops=ops,
        weights={"xshape_data": TensorConfig(shape=in_shape)},
        inputs=program_inputs,
        outputs=["output_data"])
    return program_config
