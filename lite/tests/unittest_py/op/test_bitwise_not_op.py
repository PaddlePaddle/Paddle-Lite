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
from hypothesis import given, settings, seed, example, assume, reproduce_failure
import hypothesis.strategies as st
import numpy as np
from functools import partial


class TestBitwiseNotOp(AutoScanTest):
    def __init__(self, *args, **kwargs):
        AutoScanTest.__init__(self, *args, **kwargs)
        host_places = [
            Place(TargetType.Host, PrecisionType.Any, DataLayoutType.NCHW)
        ]
        self.enable_testing_on_place(places=host_places)

    def sample_program_configs(self, draw):
        input_shape = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=64), min_size=0, max_size=4))
        input_type = draw(st.sampled_from(["int32", "int64"]))

        def generate_input(*args, **kwargs):
            if kwargs["type"] == "bool":
                return np.random.choice([True, False],
                                        kwargs["shape"]).astype(bool)
            if kwargs["type"] == "uint8":
                return np.random.normal(0, 255,
                                        kwargs["shape"]).astype(np.uint8)
            if kwargs["type"] == "int8":
                return np.random.normal(-128, 127,
                                        kwargs["shape"]).astype(np.int8)
            if kwargs["type"] == "int16":
                return np.random.normal(-32768, 32767,
                                        kwargs["shape"]).astype(np.int16)
            if kwargs["type"] == "int32":
                return np.random.normal(-2147483648, 2147483647,
                                        kwargs["shape"]).astype(np.int32)
            if kwargs["type"] == "int64":
                return np.random.normal(-9223372036854775808,
                                        9223372036854775807,
                                        kwargs["shape"]).astype(np.int64)

        bitwise_not_op = OpConfig(
            type="bitwise_not",
            inputs={"X": ["input_data"]},
            outputs={"Out": ["output_data"]},
            attrs={})

        program_config = ProgramConfig(
            ops=[bitwise_not_op],
            weights={},
            inputs={
                "input_data": TensorConfig(data_gen=partial(
                    generate_input, type=input_type, shape=input_shape)),
            },
            outputs=["output_data"])

        return program_config

    def is_program_valid(self,
                         program_config: ProgramConfig,
                         predictor_config: CxxConfig) -> bool:
        return True

    def sample_predictor_configs(self):
        return self.get_predictor_configs(), ["bitwise_not"], (1e-5, 1e-5)

    def add_ignore_pass_case(self):
        def _teller1(program_config, predictor_config):
            target_type = predictor_config.target()
            in_x_shape = list(program_config.inputs["input_data"].shape)
            if target_type not in [
                    TargetType.ARM, TargetType.Host, TargetType.X86,
                    TargetType.Metal, TargetType.OpenCL
            ]:
                if len(in_x_shape) == 0:
                    return True

        self.add_ignore_check_case(
            _teller1, IgnoreReasons.PADDLELITE_NOT_SUPPORT,
            "0D-tensor is not supported on this target now.")

    def test(self, *args, **kwargs):
        self.run_and_statis(quant=False, max_examples=200)


if __name__ == "__main__":
    unittest.main(argv=[''])
