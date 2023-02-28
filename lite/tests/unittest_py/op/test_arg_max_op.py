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
sys.path.append('../')

from auto_scan_test import AutoScanTest, IgnoreReasons
from program_config import TensorConfig, ProgramConfig, OpConfig, CxxConfig, TargetType, PrecisionType, DataLayoutType, Place
import unittest

import hypothesis
from hypothesis import given, settings, seed, example, assume
import hypothesis.strategies as st
import argparse
import numpy as np


class TestArgMaxOp(AutoScanTest):
    def __init__(self, *args, **kwargs):
        AutoScanTest.__init__(self, *args, **kwargs)
        self.enable_testing_on_place(
            TargetType.Host,
            PrecisionType.FP32,
            DataLayoutType.NCHW,
            thread=[1, 4])
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
        metal_places = [
            Place(TargetType.Metal, PrecisionType.FP32,
                  DataLayoutType.MetalTexture2DArray),
            Place(TargetType.Metal, PrecisionType.FP16,
                  DataLayoutType.MetalTexture2DArray),
            Place(TargetType.ARM, PrecisionType.FP32),
            Place(TargetType.Host, PrecisionType.FP32)
        ]
        self.enable_testing_on_place(places=metal_places)
        self.enable_testing_on_place(TargetType.NNAdapter, PrecisionType.FP32)
        self.enable_devices_on_nnadapter(device_names=[
            "nvidia_tensorrt", "intel_openvino", "kunlunxin_xtcl"
        ])

    def is_program_valid(self,
                         program_config: ProgramConfig,
                         predictor_config: CxxConfig) -> bool:
        return True

    def sample_program_configs(self, draw):
        in_shape = draw(
            st.lists(
                st.integers(
                    min_value=3, max_value=64), min_size=0, max_size=3))
        batch = draw(st.integers(min_value=1, max_value=3))
        in_shape.insert(0, batch)
        axis = draw(st.integers(min_value=-1, max_value=3))
        keepdims = draw(st.booleans())
        dtype = draw(st.sampled_from([-1, 2, 3]))
        assume(axis < len(in_shape))

        arg_max_op = OpConfig(
            type="arg_max",
            inputs={"X": ["input_data"]},
            outputs={"Out": ["output_data"]},
            attrs={
                "axis": axis,
                "keepdims": keepdims,
                "dtype": dtype,
                "flatten": False
            })
        if dtype == 2:
            arg_max_op.outputs_dtype = {"output_data": np.int32}
        else:
            arg_max_op.outputs_dtype = {"output_data": np.int64}

        program_config = ProgramConfig(
            ops=[arg_max_op],
            weights={},
            inputs={"input_data": TensorConfig(shape=in_shape)},
            outputs=["output_data"])
        return program_config

    def sample_predictor_configs(self):
        return self.get_predictor_configs(), ["arg_max"], (1e-5, 1e-5)

    def add_ignore_pass_case(self):
        def _teller1(program_config, predictor_config):
            if "nvidia_tensorrt" in self.get_nnadapter_device_name():
                set_dtype = program_config.ops[0].attrs["dtype"]
                in_shape = program_config.inputs["input_data"].shape
                axis = program_config.ops[0].attrs["axis"]
                if set_dtype != 2 or len(in_shape) == 1 or axis == 0:
                    return True

        def _teller2(program_config, predictor_config):
            set_dtype = program_config.ops[0].attrs["dtype"]
            in_shape = list(program_config.inputs["input_data"].shape)
            axis = program_config.ops[0].attrs["axis"]
            keep_dims = program_config.ops[0].attrs["keepdims"]
            if predictor_config.target() == TargetType.Metal:
                if len(in_shape) != 4 or in_shape[
                        0] != 1 or axis != 1 or keep_dims == False or set_dtype == 2:
                    return True

        def _teller3(program_config, predictor_config):
            if predictor_config.target() == TargetType.Metal:
                return True

        def _teller4(program_config, predictor_config):
            if "intel_openvino" in self.get_nnadapter_device_name():
                in_shape = program_config.inputs["input_data"].shape
                if len(in_shape) == 1:
                    return True

        def _teller5(program_config, predictor_config):
            if "kunlunxin_xtcl" in self.get_nnadapter_device_name():
                set_dtype = program_config.ops[0].attrs["dtype"]
                in_shape = program_config.inputs["input_data"].shape
                axis = program_config.ops[0].attrs["axis"]
                if len(in_shape) == 1:
                    return True

        self.add_ignore_check_case(
            _teller1, IgnoreReasons.PADDLELITE_NOT_SUPPORT,
            "Lite does not support 'int-precision output' or 'in_shape_size == 1' or 'axis == 0' on NvidiaTensorrt."
        )
        self.add_ignore_check_case(
            _teller2, IgnoreReasons.PADDLELITE_NOT_SUPPORT,
            "Lite is not supported on metal. We need to fix it as soon as possible."
        )
        self.add_ignore_check_case(
            _teller3, IgnoreReasons.ACCURACY_ERROR,
            "The op output has diff in a specific case on metal. We need to fix it as soon as possible."
        )
        self.add_ignore_check_case(
            _teller4, IgnoreReasons.PADDLELITE_NOT_SUPPORT,
            "Lite does not support 'len(in_shape) == 1' on intel OpenVINO.")

        self.add_ignore_check_case(
            _teller5, IgnoreReasons.PADDLELITE_NOT_SUPPORT,
            "Lite does not support 'in_shape_size == 1' on kunlunxin_xtcl.")

    def test(self, *args, **kwargs):
        target_str = self.get_target()
        max_examples = 100
        if target_str == "OpenCL":
            # Make sure to generate enough valid cases for OpenCL
            max_examples = 200
        if target_str == "Metal":
            max_examples = 1000
        if "kunlunxin_xtcl" in self.get_nnadapter_device_name():
            max_examples = 200
        self.run_and_statis(
            quant=False, min_success_num=25, max_examples=max_examples)


if __name__ == "__main__":
    unittest.main(argv=[''])
