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


class TestMeshgridOp(AutoScanTest):
    def __init__(self, *args, **kwargs):
        AutoScanTest.__init__(self, *args, **kwargs)
        #input tensorlist bugs will be fixed in the future
        self.enable_testing_on_place(
            TargetType.Host,
            PrecisionType.FP32,
            DataLayoutType.NCHW,
            thread=[1, 4])

    def is_program_valid(self,
                         program_config: ProgramConfig,
                         predictor_config: CxxConfig) -> bool:
        return True

    def sample_program_configs(self, draw):
        in_shape_x = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=8), min_size=4, max_size=4))
        mashgrid_op = OpConfig(
            type="meshgrid",
            inputs={"X": ["input_data_x"]},
            outputs={"Out": ["output_data"]},
            attrs={})
        program_config = ProgramConfig(
            ops=[mashgrid_op],
            weights={},
            inputs={"input_data_x": TensorConfig(shape=in_shape_x)},
            outputs=["output_data"])

        return program_config

    def sample_predictor_configs(self):
        return self.get_predictor_configs(), ["meshgrid"], (1e-5, 1e-5)

    def add_ignore_pass_case(self):
        def _teller1(program_config, predictor_config):
            if predictor_config.target() == TargetType.Host:
                return True

        self.add_ignore_check_case(
            _teller1, IgnoreReasons.PADDLE_NOT_SUPPORT,
            "Lite does not support this op in a specific case on host. We need to fix it as soon as possible."
        )

    def test(self, *args, **kwargs):
        self.run_and_statis(quant=False, max_examples=100)


if __name__ == "__main__":
    unittest.main(argv=[''])
