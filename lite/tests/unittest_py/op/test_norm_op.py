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

import numpy as np
from functools import partial
import hypothesis.strategies as st


class TestNormOp(AutoScanTest):
    def __init__(self, *args, **kwargs):
        AutoScanTest.__init__(self, *args, **kwargs)
        self.enable_testing_on_place(
            TargetType.Host,
            PrecisionType.FP32,
            DataLayoutType.NCHW,
            thread=[1, 4])
        self.enable_testing_on_place(TargetType.NNAdapter, PrecisionType.FP32)
        self.enable_devices_on_nnadapter(device_names=["cambricon_mlu"])

    def is_program_valid(self,
                         program_config: ProgramConfig,
                         predictor_config: CxxConfig) -> bool:
        x_shape = list(program_config.inputs["input_data"].shape)
        if len(x_shape) < program_config.ops[0].attrs["axis"] + 1:
            return False
        return True

    def sample_program_configs(self, draw):
        in_num = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=4), min_size=1, max_size=1))
        in_c_h_w = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=128),
                min_size=1,
                max_size=3))
        in_shape = in_num + in_c_h_w
        axis = draw(st.sampled_from([-1, 0, 1, 2, 3]))
        epsilon = draw(st.sampled_from([0.9, 1., 1.1, 1e-5]))
        norm_op = OpConfig(
            type="norm",
            inputs={"X": ["input_data"]},
            outputs={"Out": ["output_data"],
                     "Norm": ["Norm"]},
            attrs={"axis": axis,
                   "epsilon": epsilon,
                   "is_test": 1})
        program_config = ProgramConfig(
            ops=[norm_op],
            weights={},
            inputs={"input_data": TensorConfig(shape=in_shape)},
            outputs=["output_data"])
        return program_config

    def sample_predictor_configs(self):
        return self.get_predictor_configs(), ["norm"], (1e-5, 1e-5)

    def add_ignore_pass_case(self):
        pass

    def test(self, *args, **kwargs):
        self.run_and_statis(quant=False, max_examples=50)


if __name__ == "__main__":
    unittest.main(argv=[''])
