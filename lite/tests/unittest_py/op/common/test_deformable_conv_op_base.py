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
import hypothesis
from hypothesis import given, settings, seed, example, assume, reproduce_failure
import hypothesis.strategies as st


def sample_program_configs(draw):
    in_shape = draw(
        st.lists(
            st.integers(
                min_value=1, max_value=64), min_size=4, max_size=4))
    kw = np.random.randint(1, 9)
    kh = np.random.randint(1, 9)
    cout = np.random.randint(1, 128)
    cin = np.random.randint(1, 128)
    groups = draw(st.sampled_from([1, 2, cin]))
    weight_shape = [cout, cin / groups, kh, kw]
    val = in_shape[1] * groups
    assume(val == cin)
    assume(in_shape[1] == weight_shape[1])
    assume(in_shape[2] >= weight_shape[2])
    assume(in_shape[3] >= weight_shape[3])

    paddings = draw(
        st.lists(
            st.integers(
                min_value=0, max_value=2), min_size=2, max_size=2))
    dilations = draw(st.sampled_from([[1, 1]]))
    padding_algorithm = draw(st.sampled_from(["VALID", "SAME"]))
    strides = draw(st.sampled_from([[1, 1], [2, 2]]))
    im2col_step = draw(st.sampled_from([32, 64, 128]))
    data_format = "NCHW"
    deformable_groups = 1
    use_mkldnn = False

    h_out = (in_shape[2] + 2 * paddings[0] - (dilations[0] *
                                              (kh - 1) + 1) // strides[0]) + 1
    w_out = (in_shape[3] + 2 * paddings[1] - (dilations[1] *
                                              (kw - 1) + 1) // strides[1]) + 1

    def generate_input(*args, **kwargs):
        return np.random.normal(0.0, 1.0, in_shape).astype(np.float32)

    def generate_filter(*args, **kwargs):
        return np.random.normal(0.0, 1.0, weight_shape).astype(np.float32)

    def generate_bias(*args, **kwargs):
        if use_mkldnn:
            return np.random.normal(0.0, 1.0, [cout]).astype(np.float32)
        else:
            return np.zeros(shape=[cout]).astype(np.float32)

    def generate_offset(*args, **kwargs):
        offset_shape = [
            in_shape[0], deformable_groups * kw * kh * 2, h_out, w_out
        ]
        return np.random.normal(0.0, 1.0, offset_shape).astype(np.float32)

    def generate_mask(*args, **kwargs):
        mask_shape = [in_shape[0], deformable_groups * kw * kh, h_out, w_out]
        return np.random.normal(0.0, 1.0, mask_shape).astype(np.float32)

    conv_op = OpConfig(
        type="deformable_conv",
        inputs={
            "Input": ["input_data"],
            "Filter": ["filter_data"],
            "Bias": ["bias_data"],
            "Offset": ["offset_data"],
            "Mask": ["mask_data"]
        },
        outputs={"Output": ["output_data"]},
        attrs={
            "strides": strides,
            "paddings": paddings,
            "use_mkldnn": use_mkldnn,
            "padding_algorithm": padding_algorithm,
            "groups": groups,
            "deformable_groups": deformable_groups,
            "dilations": dilations,
            "data_format": data_format,
            "im2col_step": im2col_step
        })
    program_config = ProgramConfig(
        ops=[conv_op],
        weights={
            "filter_data": TensorConfig(data_gen=partial(generate_filter)),
            "bias_data": TensorConfig(data_gen=partial(generate_bias)),
            "offset_data": TensorConfig(data_gen=partial(generate_offset)),
            "mask_data": TensorConfig(data_gen=partial(generate_mask)),
        },
        inputs={"input_data": TensorConfig(data_gen=partial(generate_input))},
        outputs=["output_data"])
    return program_config
