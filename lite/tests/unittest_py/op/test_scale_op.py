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
import numpy as np
from functools import partial
import argparse


class TestScaleOp(AutoScanTest):
    def __init__(self, *args, **kwargs):
        AutoScanTest.__init__(self, *args, **kwargs)
        self.enable_testing_on_place(
            TargetType.ARM,
            PrecisionType.FP32,
            DataLayoutType.NCHW,
            thread=[1, 4])
        self.enable_testing_on_place(
            TargetType.X86,
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
        self.enable_testing_on_place(
            TargetType.ARM,
            PrecisionType.FP16,
            DataLayoutType.NCHW,
            thread=[1, 4])
        self.enable_testing_on_place(TargetType.NNAdapter, PrecisionType.FP32)
        self.enable_devices_on_nnadapter(device_names=[
            "kunlunxin_xtcl", "cambricon_mlu", "nvidia_tensorrt",
            "intel_openvino"
        ])

    def is_program_valid(self,
                         program_config: ProgramConfig,
                         predictor_config: CxxConfig) -> bool:
        x_dtype = program_config.inputs["input_data"].dtype
        target_type = predictor_config.target()
        if target_type in [TargetType.ARM]:
            if predictor_config.precision(
            ) == PrecisionType.FP16 and x_dtype != np.float32:
                return False
        if target_type == TargetType.NNAdapter:
            if program_config.inputs["input_data"].dtype != np.float32:
                return False
        return True

    def sample_program_configs(self, draw):
        in_shape = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=8), min_size=1, max_size=4))
        bias = draw(st.floats(min_value=-5, max_value=5))
        bias_after_scale = draw(st.booleans())
        scale = draw(st.floats(min_value=-5, max_value=5))
        input_type = draw(st.sampled_from(["int32", "int64", "float32"]))
        has_scale_tensor = False  # draw(st.booleans())

        def generate_data(*args, **kwargs):
            low, high = -10, 10
            dtype = "float32"
            shape = kwargs["shape"]
            if "low" in kwargs:
                low = kwargs["low"]
            if "high" in kwargs:
                high = kwargs["high"]
            if "dtype" in kwargs:
                dtype = kwargs["dtype"]

            if dtype == "int32":
                if low == high:
                    return low * np.ones(shape).astype(np.int32)
                else:
                    return np.random.randint(low, high, shape).astype(np.int32)
            elif dtype == "int64":
                if low == high:
                    return low * np.ones(shape).astype(np.int64)
                else:
                    return np.random.randint(low, high, shape).astype(np.int64)
            elif dtype == "float32":
                return (high - low
                        ) * np.random.random(shape).astype(np.float32) + low

        input_dict = {"X": ["input_data"]}
        input_data_dict = {
            "input_data": TensorConfig(data_gen=partial(
                generate_data, dtype=input_type, shape=in_shape))
        }
        if has_scale_tensor:
            input_dict["ScaleTensor"] = "scale_tensor_data"
            input_data_dict["scale_tensor_data"] = TensorConfig(shape=[1, ])

        scale_op = OpConfig(
            type="scale",
            inputs=input_dict,
            outputs={"Out": ["output_data"]},
            attrs={
                "bias": bias,
                "bias_after_scale": bias_after_scale,
                "scale": scale
            })

        program_config = ProgramConfig(
            ops=[scale_op],
            weights={},
            inputs=input_data_dict,
            outputs=["output_data"])

        return program_config

    def sample_predictor_configs(self):
        atol, rtol = 1e-5, 1e-5
        target_str = self.get_target()
        if target_str == "Metal":
            atol, rtol = 1e-2, 1e-2
        return self.get_predictor_configs(), ["scale"], (atol, rtol)

    def add_ignore_pass_case(self):
        def teller1(program_config, predictor_config):
            target_type = predictor_config.target()
            in_shape = list(program_config.inputs["input_data"].shape)
            in_data_type = program_config.inputs["input_data"].dtype
            if target_type == TargetType.Metal:
                if len(in_shape) != 4 or in_data_type != "float32":
                    return True

        def teller2(program_config, predictor_config):
            target_type = predictor_config.target()
            if target_type == TargetType.Metal:
                return True

        def teller3(program_config, predictor_config):
            target_type = predictor_config.target()
            x_dtype = program_config.inputs["input_data"].dtype
            if target_type == TargetType.OpenCL:
                if x_dtype == np.int32 or x_dtype == np.int64:
                    return True

        def teller4(program_config, predictor_config):
            if "nvidia_tensorrt" in self.get_nnadapter_device_name():
                in_shape = program_config.inputs["input_data"].shape
                if len(in_shape) == 1:
                    return True

        self.add_ignore_check_case(
            teller1, IgnoreReasons.PADDLELITE_NOT_SUPPORT,
            "Lite does not support this op in a specific case. We need to fix it as soon as possible."
        )
        self.add_ignore_check_case(
            teller2, IgnoreReasons.ACCURACY_ERROR,
            "The op output has diff in a specific case on metal. We need to fix it as soon as possible."
        )
        self.add_ignore_check_case(
            teller3, IgnoreReasons.PADDLELITE_NOT_SUPPORT,
            "Lite does not support this op when dtype is int32 or int64 on Opencl. "
        )
        self.add_ignore_check_case(
            teller4, IgnoreReasons.PADDLELITE_NOT_SUPPORT,
            "Lite does not support 'in_shape_size == 1' on nvidia_tensorrt.")

    def test(self, *args, **kwargs):
        target_str = self.get_target()
        max_examples = 100
        if target_str in ["OpenCL", "Metal"]:
            # Make sure to generate enough valid cases for specific targets
            max_examples = 2000
        elif target_str in ["NNAdapter"]:
            # Make sure to generate enough valid cases for specific targets
            max_examples = 300
        self.run_and_statis(
            quant=False, min_success_num=25, max_examples=max_examples)


if __name__ == "__main__":
    unittest.main(argv=[''])
