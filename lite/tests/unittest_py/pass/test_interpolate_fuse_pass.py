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

from auto_scan_test import FusePassAutoScanTest
from program_config import TensorConfig, ProgramConfig, OpConfig, CxxConfig, TargetType, PrecisionType, DataLayoutType, Place
import numpy as np
from functools import partial
from typing import Optional, List, Callable, Dict, Any, Set
import unittest
import math

import hypothesis
from hypothesis import given, settings, seed, example, assume, reproduce_failure
import hypothesis.strategies as st


def sample_program_configs(draw, interp_type):
    interpolate_type = draw(
        st.sampled_from(['interpolate_type_1', 'interpolate_type_2']))

    #shape params
    batch = draw(st.integers(min_value=1, max_value=2))
    channel = draw(st.integers(min_value=1, max_value=32))
    height = draw(st.integers(min_value=3, max_value=100))
    width = draw(st.integers(min_value=3, max_value=100))
    shape_op_input_shape = [batch, channel, height, width]
    #slice params
    axes = [0]
    starts = [2]
    ends = [4]
    infer_flags = [1]
    #cast params
    cast_op_in_dtype = draw(st.sampled_from([2, 3]))
    cast_op_out_dtype = draw(st.sampled_from([2]))

    if interpolate_type == "interpolate_type_1":
        #fill_constant_op params
        fill_constant_in_shape = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=2), min_size=1, max_size=1))
        fill_constant_value = draw(st.integers(min_value=1, max_value=16))
        #bilinear_interp params
        scale = draw(st.integers(min_value=1, max_value=16))

        if interp_type == "bilinear_interp":
            interp_method = "bilinear"
            align_corners = draw(st.booleans())
            align_mode = draw(st.sampled_from([0, 1]))
        else:
            interp_method = "nearest"
            align_corners = False
            align_mode = 0

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
            inputs={},
            outputs={"Out": ["fill_constant_output"]},
            attrs={
                "dtype": cast_op_out_dtype,
                "shape": fill_constant_in_shape,
                "value":
                fill_constant_value,  #The value of the floating point number must be an integer, otherwise out has diff
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
                    "OutSize": ["elementwise_mul_out"]},
            outputs={"Out": ["output_data"]},
            attrs={
                "data_layout": "NCHW",
                "scale": scale,
                "interp_method": interp_method,
                "align_corners": align_corners,
                "align_mode": align_mode
            })

        ops = [
            shape_op, slice_op, cast_op, fill_constant_op, elementwise_mul_op,
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
        #scale op params
        scale = draw(st.integers(
            min_value=1, max_value=10))  #floats output shapes are not equal
        bias = draw(st.floats(
            min_value=0, max_value=1))  #>1 output shapes are not equal
        bias_after_scale = draw(st.sampled_from([False]))  #required in pass
        #bilinear_interp params
        if interp_type == "bilinear_interp":
            interp_method = "bilinear"
            align_corners = draw(st.booleans())
            align_mode = draw(st.sampled_from([0, 1]))
        else:
            interp_method = "nearest"
            align_corners = False
            align_mode = 0

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
                    "OutSize": ["scale_output"]},
            outputs={"Out": ["output_data"]},
            attrs={
                "data_layout": "NCHW",
                "scale": scale,
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


class TestInterpolateBilinearFusePass(FusePassAutoScanTest):
    def __init__(self, *args, **kwargs):
        FusePassAutoScanTest.__init__(self, *args, **kwargs)
        self.enable_testing_on_place(
            TargetType.ARM,
            PrecisionType.FP32,
            DataLayoutType.NCHW,
            thread=[1, 4])
        #opencl
        opencl_places = [
            Place(TargetType.OpenCL, PrecisionType.FP16,
                  DataLayoutType.ImageDefault), Place(
                      TargetType.OpenCL, PrecisionType.FP16,
                      DataLayoutType.ImageFolder),
            Place(TargetType.OpenCL, PrecisionType.FP32, DataLayoutType.NCHW),
            Place(TargetType.OpenCL, PrecisionType.Any,
                  DataLayoutType.ImageDefault), Place(
                      TargetType.OpenCL, PrecisionType.Any,
                      DataLayoutType.ImageFolder),
            Place(TargetType.OpenCL, PrecisionType.Any, DataLayoutType.NCHW),
            Place(TargetType.Host, PrecisionType.FP32)
        ]
        self.enable_testing_on_place(places=opencl_places)
        #x86
        self.enable_testing_on_place(
            TargetType.X86,
            PrecisionType.FP32,
            DataLayoutType.NCHW,
            thread=[1, 4])
        # metal
        metal_places = [
            Place(TargetType.Metal, PrecisionType.FP32,
                  DataLayoutType.MetalTexture2DArray),
            Place(TargetType.Metal, PrecisionType.FP16,
                  DataLayoutType.MetalTexture2DArray),
            Place(TargetType.ARM, PrecisionType.FP32),
            Place(TargetType.Host, PrecisionType.FP32)
        ]
        # self.enable_testing_on_place(places=metal_places)

    def is_program_valid(self,
                         program_config: ProgramConfig,
                         predictor_config: CxxConfig) -> bool:
        return True

    def sample_program_configs(self, draw):
        return sample_program_configs(draw, interp_type="bilinear_interp")

    def sample_predictor_configs(self):
        return self.get_predictor_configs(), ['bilinear_interp'], (1e-5, 1e-5)

    def add_ignore_pass_case(self):
        pass

    def test(self, *args, **kwargs):
        self.run_and_statis(
            quant=False,
            max_examples=100,
            passes=["lite_interpolate_fuse_pass"])


class TestInterpolateNearestFusePass(FusePassAutoScanTest):
    def __init__(self, *args, **kwargs):
        FusePassAutoScanTest.__init__(self, *args, **kwargs)
        self.enable_testing_on_place(
            TargetType.ARM,
            PrecisionType.FP32,
            DataLayoutType.NCHW,
            thread=[1, 4])
        #opencl
        opencl_places = [
            Place(TargetType.OpenCL, PrecisionType.FP16,
                  DataLayoutType.ImageDefault), Place(
                      TargetType.OpenCL, PrecisionType.FP16,
                      DataLayoutType.ImageFolder),
            Place(TargetType.OpenCL, PrecisionType.FP32, DataLayoutType.NCHW),
            Place(TargetType.OpenCL, PrecisionType.Any,
                  DataLayoutType.ImageDefault), Place(
                      TargetType.OpenCL, PrecisionType.Any,
                      DataLayoutType.ImageFolder),
            Place(TargetType.OpenCL, PrecisionType.Any, DataLayoutType.NCHW),
            Place(TargetType.Host, PrecisionType.FP32)
        ]
        self.enable_testing_on_place(places=opencl_places)
        #x86
        # self.enable_testing_on_place(
        #     TargetType.X86,
        #     PrecisionType.FP32,
        #     DataLayoutType.NCHW,
        #     thread=[1, 4])
        # metal
        metal_places = [
            Place(TargetType.Metal, PrecisionType.FP32,
                  DataLayoutType.MetalTexture2DArray),
            Place(TargetType.Metal, PrecisionType.FP16,
                  DataLayoutType.MetalTexture2DArray),
            Place(TargetType.ARM, PrecisionType.FP32),
            Place(TargetType.Host, PrecisionType.FP32)
        ]
        # self.enable_testing_on_place(places=metal_places)

    def is_program_valid(self,
                         program_config: ProgramConfig,
                         predictor_config: CxxConfig) -> bool:
        return True

    def sample_program_configs(self, draw):
        return sample_program_configs(draw, interp_type="nearest_interp")

    def sample_predictor_configs(self):
        return self.get_predictor_configs(), ['nearest_interp'], (1e-5, 1e-5)

    def add_ignore_pass_case(self):
        pass

    def test(self, *args, **kwargs):
        self.run_and_statis(
            quant=False,
            max_examples=100,
            passes=["lite_interpolate_fuse_pass"])


if __name__ == "__main__":
    unittest.main(argv=[''])
