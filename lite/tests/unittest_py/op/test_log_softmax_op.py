# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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


class TestLogSoftmaxOp(AutoScanTest):
    def __init__(self, *args, **kwargs):
        AutoScanTest.__init__(self, *args, **kwargs)
        host_places = [
            Place(TargetType.Host, PrecisionType.FP32, DataLayoutType.NCHW)
        ]
        self.enable_testing_on_place(places=host_places, thread=[1, 4])
        self.enable_testing_on_place(TargetType.NNAdapter, PrecisionType.FP32)
        self.enable_devices_on_nnadapter(device_names=["nvidia_tensorrt"])

    def is_program_valid(self,
                         program_config: ProgramConfig,
                         predictor_config: CxxConfig) -> bool:
        return True

    def sample_program_configs(self, draw):
        in_shape = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=64), min_size=0, max_size=3))
        in_shape.insert(0, draw(st.integers(min_value=1, max_value=4)))
        input_axis = draw(st.sampled_from([0, 1, 2, 3, -1]))
        assume(len(in_shape) > 1 and input_axis < len(in_shape))

        def generate_input(*args, **kwargs):
            return np.random.normal(0.0, 1.0, in_shape).astype(np.float32)

        ops_config = OpConfig(
            type="log_softmax",
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
        if target_str == "NNAdapter":
            atol, rtol = 4e-5, 4e-5
        return self.get_predictor_configs(), ["log_softmax"], (atol, rtol)

    def add_ignore_pass_case(self):
        def teller1(program_config, predictor_config):
            if self.get_nnadapter_device_name() == "nvidia_tensorrt":
                in_shape = program_config.inputs["input_data"].shape
                axis = program_config.ops[0].attrs["axis"]
                if len(in_shape) < 2 or axis == 0 or axis == -len(in_shape):
                    return True

        self.add_ignore_check_case(
            teller1, IgnoreReasons.PADDLELITE_NOT_SUPPORT,
            "Lite does not support 'in_shape_size == 1' or 'axis == 0' on nvidia_tensorrt."
        )

    def test(self, *args, **kwargs):
        max_examples = 100
        self.run_and_statis(quant=False, max_examples=max_examples)


if __name__ == "__main__":
    unittest.main(argv=[''])
