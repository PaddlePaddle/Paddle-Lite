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
from functools import partial
import hypothesis
from hypothesis import given, settings, seed, example, assume, reproduce_failure
import hypothesis.strategies as st
import numpy as np


class TestSoftmaxOp(AutoScanTest):
    def __init__(self, *args, **kwargs):
        AutoScanTest.__init__(self, *args, **kwargs)
        self.enable_testing_on_place(
            TargetType.X86, [PrecisionType.FP32],
            DataLayoutType.NCHW,
            thread=[1, 4])
        self.enable_testing_on_place(
            TargetType.ARM, [PrecisionType.FP32],
            DataLayoutType.NCHW,
            thread=[1, 4])
        opencl_places = [
            Place(TargetType.OpenCL, PrecisionType.FP16,
                  DataLayoutType.ImageFolder), Place(
                      TargetType.OpenCL, PrecisionType.FP16,
                      DataLayoutType.ImageDefault),
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
        self.enable_testing_on_place(
            TargetType.ARM,
            PrecisionType.FP16,
            DataLayoutType.NCHW,
            thread=[1, 4])
        self.enable_devices_on_nnadapter(device_names=[
            "kunlunxin_xtcl", "cambricon_mlu", "nvidia_tensorrt"
        ])

    def is_program_valid(self,
                         program_config: ProgramConfig,
                         predictor_config: CxxConfig) -> bool:
        x_shape = list(program_config.inputs["input_data"].shape)
        axis = program_config.ops[0].attrs["axis"]
        if predictor_config.target() == TargetType.Metal:
            if len(x_shape) != 4 or axis != 1 or x_shape[0] != 1:
                return False
        if predictor_config.target() == TargetType.NNAdapter:
            if "nvidia_tensorrt" in self.get_nnadapter_device_name():
                if len(x_shape) < 2:
                    return False
            if "kunlunxin_xtcl" in self.get_nnadapter_device_name():
                if axis == 0:
                    return False
        return True

    def sample_program_configs(self, draw):
        in_shape = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=64), min_size=0, max_size=3))
        in_shape.insert(0, draw(st.integers(min_value=1, max_value=10)))
        input_axis = draw(st.sampled_from([0, 1, 2, 3, -1]))
        assume(len(in_shape) > 1 and input_axis < len(in_shape))

        in_shape = draw(st.sampled_from([in_shape, []]))
        if in_shape == []:
            input_axis = -1

        def generate_input(*args, **kwargs):
            return np.random.random(in_shape).astype(np.float32)

        ops_config = OpConfig(
            type="softmax",
            inputs={"X": ["input_data"]},
            outputs={"Out": ["output_data"]},
            attrs={"axis": input_axis})

        program_config = ProgramConfig(
            ops=[ops_config],
            weights={},
            inputs={
                "input_data": TensorConfig(data_gen=partial(generate_input))
            },
            outputs=["output_data"])

        return program_config

    def sample_predictor_configs(self):
        atol, rtol = 1e-5, 1e-5
        target_str = self.get_target()
        if target_str == "Metal":
            atol, rtol = 1e-3, 1e-3
        elif target_str == "OpenCL":
            atol, rtol = 1e-4, 1e-4
        elif target_str == "NNAdapter":
            atol, rtol = 4e-5, 4e-5
        return self.get_predictor_configs(), ["softmax"], (atol, rtol)

    def add_ignore_pass_case(self):
        def teller1(program_config, predictor_config):
            target_type = predictor_config.target()
            if target_type == TargetType.OpenCL or target_type == TargetType.X86:
                return True

        def teller3(program_config, predictor_config):
            if self.get_nnadapter_device_name() == "nvidia_tensorrt":
                in_shape = program_config.inputs["input_data"].shape
                axis = program_config.ops[0].attrs["axis"]
                if len(in_shape) == 1 or axis == 0 or axis == -len(in_shape):
                    return True

        def teller4(program_config, predictor_config):
            target_type = predictor_config.target()
            in_x_shape = list(program_config.inputs["input_data"].shape)
            if target_type not in [
                    TargetType.ARM, TargetType.Host, TargetType.OpenCL,
                    TargetType.Metal, TargetType.X86
            ]:
                if len(in_x_shape) == 0:
                    return True

        def teller5(program_config, predictor_config):
            precision_type = predictor_config.precision()
            target_type = predictor_config.target()
            if precision_type == PrecisionType.FP16 and target_type == TargetType.ARM:
                return True

        self.add_ignore_check_case(
            teller1, IgnoreReasons.PADDLELITE_NOT_SUPPORT,
            "OpenCL has diff and doesn't support 0D, X86 has diff")
        self.add_ignore_check_case(
            teller3, IgnoreReasons.PADDLELITE_NOT_SUPPORT,
            "Lite does not support 'in_shape_size == 1' or 'axis == 0' on nvidia_tensorrt."
        )
        self.add_ignore_check_case(teller4,
                                   IgnoreReasons.PADDLELITE_NOT_SUPPORT,
                                   "Only test 0D-tensor on CPU(ARM/Host) now.")
        self.add_ignore_check_case(teller5, IgnoreReasons.ACCURACY_ERROR,
                                   "ARM FP16 has diff.")

    def test(self, *args, **kwargs):
        target_str = self.get_target()
        max_examples = 100
        if target_str == "OpenCL":
            # Make sure to generate enough valid cases for OpenCL
            max_examples = 100
        elif target_str == "Metal":
            # Make sure to generate enough valid cases for Metal
            max_examples = 2500
        elif target_str == "NNAdapter":
            # Make sure to generate enough valid cases for NNAdapter
            max_examples = 200
        self.run_and_statis(quant=False, max_examples=max_examples)


if __name__ == "__main__":
    unittest.main(argv=[''])
