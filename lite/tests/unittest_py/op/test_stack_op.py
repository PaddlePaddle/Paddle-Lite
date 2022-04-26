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
from hypothesis import given, settings, seed, example, assume
import hypothesis.strategies as st
import numpy as np


class TestStackOp(AutoScanTest):
    def __init__(self, *args, **kwargs):
        AutoScanTest.__init__(self, *args, **kwargs)
        self.enable_testing_on_place(
            TargetType.Host, [PrecisionType.FP32],
            DataLayoutType.NCHW,
            thread=[1, 4])
        self.enable_testing_on_place(TargetType.NNAdapter, PrecisionType.FP32)
        self.enable_devices_on_nnadapter(
            device_names=["nvidia_tensorrt", "intel_openvino"])

    def is_program_valid(self,
                         program_config: ProgramConfig,
                         predictor_config: CxxConfig) -> bool:
        return True

    def sample_program_configs(self, draw):
        in_shape = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=64), min_size=1, max_size=4))
        input_type = draw(st.sampled_from(["float32", "int64", "int32"]))
        input_axis = draw(st.sampled_from([-1, 0, 1, 2, 3]))
        assume(input_axis >= -(len(in_shape) + 1))
        assume(input_axis < (len(in_shape) + 1))
        target_str = self.get_target()
        if target_str == "NNAdapter":
            assume(input_type != "int64")

        def generate_input1(*args, **kwargs):
            if input_type == "float32":
                return np.random.normal(1.0, 6.0, in_shape).astype(np.float32)
            elif input_type == "int32":
                return np.random.normal(1.0, 6.0, in_shape).astype(np.int32)
            elif input_type == "int64":
                return np.random.normal(1.0, 6.0, in_shape).astype(np.int64)

        def generate_input2(*args, **kwargs):
            if input_type == "float32":
                return np.random.normal(1.0, 6.0, in_shape).astype(np.float32)
            elif input_type == "int32":
                return np.random.normal(1.0, 6.0, in_shape).astype(np.int32)
            elif input_type == "int64":
                return np.random.normal(1.0, 6.0, in_shape).astype(np.int64)

        def generate_input3(*args, **kwargs):
            if input_type == "float32":
                return np.random.normal(1.0, 6.0, in_shape).astype(np.float32)
            elif input_type == "int32":
                return np.random.normal(1.0, 6.0, in_shape).astype(np.int32)
            elif input_type == "int64":
                return np.random.normal(1.0, 6.0, in_shape).astype(np.int64)

        ops_config = OpConfig(
            type="stack",
            inputs={"X": ["stack_input1", "stack_input2", "stack_input3"]},
            outputs={"Y": ["output_data"]},
            attrs={"axis": input_axis})

        program_config = ProgramConfig(
            ops=[ops_config],
            weights={},
            inputs={
                "stack_input1":
                TensorConfig(data_gen=partial(generate_input1)),
                "stack_input2":
                TensorConfig(data_gen=partial(generate_input2)),
                "stack_input3": TensorConfig(data_gen=partial(generate_input3))
            },
            outputs=["output_data"])

        return program_config

    def sample_predictor_configs(self):
        return self.get_predictor_configs(), ["stack"], (1e-5, 1e-5)

    def add_ignore_pass_case(self):
        def teller1(program_config, predictor_config):
            if "nvidia_tensorrt" in self.get_nnadapter_device_name():
                in_shape = program_config.inputs["stack_input1"].shape
                axis = program_config.ops[0].attrs["axis"]
                if len(in_shape) == 1 or axis == 0:
                    return True

        self.add_ignore_check_case(
            teller1, IgnoreReasons.PADDLELITE_NOT_SUPPORT,
            "Lite does not support 'in_shape_size == 1' or 'axis == 0' on NvidiaTensorrt."
        )

    def test(self, *args, **kwargs):
        self.run_and_statis(quant=False, max_examples=100)


if __name__ == "__main__":
    unittest.main(argv=[''])
