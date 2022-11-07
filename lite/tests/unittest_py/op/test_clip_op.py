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
from functools import partial


class TestClipOp(AutoScanTest):
    def __init__(self, *args, **kwargs):
        AutoScanTest.__init__(self, *args, **kwargs)
        self.enable_testing_on_place(
            TargetType.X86,
            PrecisionType.FP32,
            DataLayoutType.NCHW,
            thread=[1, 4])
        self.enable_testing_on_place(
            TargetType.ARM,
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
        self.enable_testing_on_place(TargetType.NNAdapter, PrecisionType.FP32)
        self.enable_devices_on_nnadapter(device_names=[
            "cambricon_mlu", "nvidia_tensorrt", "intel_openvino",
            "kunlunxin_xtcl"
        ])

    def is_program_valid(self,
                         program_config: ProgramConfig,
                         predictor_config: CxxConfig) -> bool:
        if "kunlunxin_xtcl" in self.get_nnadapter_device_name():
            in_num = len(program_config.inputs)
            if in_num == 3:
                return False
        return True

    def sample_program_configs(self, draw):
        in_shape = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=8), min_size=1, max_size=4))
        min_val = float(np.random.randint(0, 100) / 100)
        max_val = min_val + 0.5
        min_max_shape = draw(st.integers(min_value=1, max_value=20))
        case_type = draw(st.sampled_from(["c1", "c2"]))

        def generate_input(*args, **kwargs):
            return np.random.random(in_shape).astype(np.float32)

        def generate_min(*args, **kwargs):
            return np.random.random(min_max_shape).astype(np.float32)

        def generate_max(*args, **kwargs):
            return np.random.random(min_max_shape).astype(np.float32) + 1.0

        if case_type == "c1":
            clip_op = OpConfig(
                type="clip",
                inputs={
                    "X": ["input_data"],
                    "Min": ["min_data"],
                    "Max": ["max_data"]
                },
                outputs={"Out": ["output_data"]},
                attrs={"min": 0,
                       "max": 0})
            program_config = ProgramConfig(
                ops=[clip_op],
                weights={},
                inputs={
                    "input_data":
                    TensorConfig(data_gen=partial(generate_input)),
                    "min_data": TensorConfig(data_gen=partial(generate_min)),
                    "max_data": TensorConfig(data_gen=partial(generate_max)),
                },
                outputs=["output_data"])
        else:
            clip_op = OpConfig(
                type="clip",
                inputs={"X": ["input_data"]},
                outputs={"Out": ["output_data"]},
                attrs={"min": min_val,
                       "max": max_val})
            program_config = ProgramConfig(
                ops=[clip_op],
                weights={},
                inputs={
                    "input_data":
                    TensorConfig(data_gen=partial(generate_input))
                },
                outputs=["output_data"])
        return program_config

    def sample_predictor_configs(self):
        return self.get_predictor_configs(), ["clip"], (1e-5, 1e-5)

    def add_ignore_pass_case(self):
        def teller1(program_config, predictor_config):
            if "nvidia_tensorrt" in self.get_nnadapter_device_name():
                in_num = len(program_config.inputs)
                in_shape_size = len(program_config.inputs["input_data"].shape)
                if in_num == 3 or in_shape_size == 1:
                    return True

        self.add_ignore_check_case(
            teller1, IgnoreReasons.PADDLELITE_NOT_SUPPORT,
            "Lite does not support '3 input tensors' or 'in_shape_size == 1' on nvidia_tensorrt."
        )

        def teller2(program_config, predictor_config):
            if "intel_openvino" in self.get_nnadapter_device_name():
                in_num = len(program_config.inputs)
                if in_num == 3:
                    return True

        self.add_ignore_check_case(
            teller2, IgnoreReasons.PADDLELITE_NOT_SUPPORT,
            "Lite does not support '3 input tensors' on intel_openvino.")

    def test(self, *args, **kwargs):
        self.run_and_statis(quant=False, max_examples=100)


if __name__ == "__main__":
    unittest.main(argv=[''])
