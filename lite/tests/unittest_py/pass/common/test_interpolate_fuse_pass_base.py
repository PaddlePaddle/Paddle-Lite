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


def sample_program_configs(draw, interp_type):
    interpolate_type = draw(
        st.sampled_from(['interpolate_type_1', 'interpolate_type_2']))
    if interpolate_type == "interpolate_type_1":
        #shape params
        shape_op_input_shape = draw(
            st.lists(
                st.integers(
                    min_value=2, max_value=10), min_size=4,
                max_size=4))  #min_value=1, Output has nan
        #slice params
        axes = [0]
        starts = [2]
        ends = [4]
        infer_flags = [1]
        #cast params
        cast_op_in_dtype = draw(st.sampled_from([2, 3]))  #float type has diff
        cast_op_out_dtype = draw(st.sampled_from(
            [2]))  #bilinear_interp's OutSize must be int
        #fill_constant_op params
        fill_constant_in_shape = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=2), min_size=1, max_size=1))
        fill_constant_value = draw(st.integers(
            min_value=3, max_value=3))  #1 2 3 is ok, other has diff
        #bilinear_interp params
        scale = draw(st.floats(min_value=0.1, max_value=0.9))
        align_corners = draw(st.booleans())
        align_mode = draw(st.sampled_from([0, 1]))
        if interp_type == "bilinear_interp":
            interp_method = "bilinear"
        else:
            interp_method = "nearest"

        def generate_input(*args, **kwargs):
            return np.random.random(shape_op_input_shape).astype(np.float32)

        shape_op = OpConfig(
            type="shape",
            inputs={"Input": ["input_data"]},
            outputs={"Out": ["shape_out"]},
            attrs={})

        slice_op = OpConfig(
            type="slice",
            inputs={"Input": ["shape_out"]},
            outputs={"Out": ["slice_out"]},
            attrs={
                "axes": axes,
                "starts": starts,
                "ends": ends,
                "decrease_axis": [],
                "infer_flags": infer_flags
            })

        cast_op = OpConfig(
            type="cast",
            inputs={"X": ["slice_out"]},
            outputs={"Out": ["cast_out"]},
            attrs={
                "in_dtype": cast_op_in_dtype,
                "out_dtype": cast_op_out_dtype
            })

        fill_constant_op = OpConfig(
            type="fill_constant",
            inputs={},  #Only when the input is empty can the fusion succeed
            outputs={"Out": ["fill_constant_output"]},
            attrs={
                "dtype": cast_op_out_dtype,
                "shape": fill_constant_in_shape,
                "value": float(
                    fill_constant_value
                ),  #The value of the floating point number must be an integer, otherwise out has diff
                "force_cpu": False
            })

        elementwise_mul_op = OpConfig(
            type="elementwise_mul",
            inputs={"X": ["fill_constant_output"],
                    "Y": ["cast_out"]},
            outputs={"Out": ["elementwise_mul_out"]},
            attrs={"axis": -1})

        interpolate_op = OpConfig(
            type=interp_type,
            inputs={"X": ["input_data"],
                    "OutSize": ["elementwise_mul_out"]
                    },  #OutSize's dimension[0] must be 2
            outputs={"Out": ["output_data"]},
            attrs={
                "data_layout": "NCHW",
                "scale": scale,
                "interp_method": interp_method,
                "align_corners": align_corners,
                "align_mode": align_mode
            })

        ops = [
            fill_constant_op, shape_op, slice_op, cast_op, elementwise_mul_op,
            interpolate_op
        ]
        program_config = ProgramConfig(
            ops=ops,
            weights={},
            inputs={
                "input_data": TensorConfig(data_gen=partial(generate_input))
            },
            outputs=["output_data"])
        return program_config

    else:
        #shape params
        shape_op_input_shape = draw(
            st.lists(
                st.integers(
                    min_value=2, max_value=10), min_size=4,
                max_size=4))  #min_value=1, Output has nan
        #slice params
        axes = [0]
        starts = [2]
        ends = [4]
        infer_flags = [1]
        #cast params
        cast_op_in_dtype = draw(st.sampled_from([2, 3]))  #float type has diff
        cast_op_out_dtype = draw(st.sampled_from(
            [2]))  #bilinear_interp's OutSize must be int
        #scale op params
        scale = draw(st.floats(
            min_value=3, max_value=3))  #1 2 3 is ok, other has diff
        bias = draw(st.floats(min_value=0, max_value=1))
        bias_after_scale = draw(st.sampled_from([False]))  #required in pass
        #bilinear_interp params
        # scale = draw(st.floats(min_value=0.1, max_value=0.9))
        align_corners = draw(st.booleans())
        align_mode = draw(st.sampled_from([0, 1]))
        if interp_type == "bilinear_interp":
            interp_method = "bilinear"
        else:
            interp_method = "nearest"

        def generate_input(*args, **kwargs):
            return np.random.random(shape_op_input_shape).astype(np.float32)

        shape_op = OpConfig(
            type="shape",
            inputs={"Input": ["input_data"]},
            outputs={"Out": ["shape_out"]},
            attrs={})

        slice_op = OpConfig(
            type="slice",
            inputs={"Input": ["shape_out"]},
            outputs={"Out": ["slice_out"]},
            attrs={
                "axes": axes,
                "starts": starts,
                "ends": ends,
                "decrease_axis": [],
                "infer_flags": infer_flags
            })

        cast_op = OpConfig(
            type="cast",
            inputs={"X": ["slice_out"]},
            outputs={"Out": ["cast_out"]},
            attrs={
                "in_dtype": cast_op_in_dtype,
                "out_dtype": cast_op_out_dtype
            })

        scale_op = OpConfig(
            type="scale",
            inputs={"X": ["cast_out"]},
            outputs={"Out": ["scale_output"]},
            attrs={
                "scale": scale,
                "bias": bias,
                "bias_after_scale": bias_after_scale
            })

        interpolate_op = OpConfig(
            type=interp_type,
            inputs={"X": ["input_data"],
                    "OutSize":
                    ["scale_output"]},  #OutSize's dimension[0] must be 2
            outputs={"Out": ["output_data"]},
            attrs={
                "data_layout": "NCHW",
                "scale":
                scale,  #only interpolate_scale == scale_op_scale, out is ok, else out has diff
                "interp_method": interp_method,
                "align_corners": align_corners,
                "align_mode": align_mode
            })

        ops = [shape_op, slice_op, cast_op, scale_op, interpolate_op]
        program_config = ProgramConfig(
            ops=ops,
            weights={},
            inputs={
                "input_data": TensorConfig(data_gen=partial(generate_input))
            },
            outputs=["output_data"])
        return program_config
