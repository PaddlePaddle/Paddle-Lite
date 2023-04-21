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
from functools import partial
import numpy as np


class TestAffineGridOp(AutoScanTest):
    def __init__(self, *args, **kwargs):
        AutoScanTest.__init__(self, *args, **kwargs)
        # only support arm
        self.enable_testing_on_place(
            TargetType.ARM,
            PrecisionType.FP32,
            DataLayoutType.NCHW,
            thread=[1, 4])

    def is_program_valid(self,
                         program_config: ProgramConfig,
                         predictor_config: CxxConfig) -> bool:
        return True

    def sample_program_configs(self, draw):
        in_shape = [draw(st.integers(min_value=1, max_value=50)), 2, 3]
        align_corners = draw(st.booleans())
        output_shape = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=100),
                min_size=4,
                max_size=4))

        def generate_input(*args, **kwargs):
            return np.random.random(in_shape).astype(np.float32)

        def generate_output_shape(*args, **kwargs):
            return np.random.random([]).astype(np.int32)

        affine_grid_op = OpConfig(
            type="affine_grid",
            inputs={
                "Theta": ["input_data"],
                "OutputShape": ["output_shape_data"]
            },
            outputs={"Output": ["output_data"]},
            attrs={
                "output_shape": output_shape,
                "align_corners": align_corners
            })
        program_config = ProgramConfig(
            ops=[affine_grid_op],
            weights={},
            inputs={
                "input_data": TensorConfig(data_gen=partial(generate_input)),
                "output_shape_data":
                TensorConfig(data_gen=partial(generate_output_shape))
            },
            outputs=["output_data"])
        return program_config

    def sample_predictor_configs(self):
        return self.get_predictor_configs(), ["affine_grid"], (1e-5, 1e-5)

    def add_ignore_pass_case(self):
        def _teller1(program_config, predictor_config):
            target_type = predictor_config.target()
            if target_type == TargetType.ARM:
                return True

        self.add_ignore_check_case(_teller1, IgnoreReasons.PADDLE_NOT_SUPPORT,
                                   "paddle(ARM) report Fail to alloc memory.")

    def test(self, *args, **kwargs):
        self.run_and_statis(quant=False, max_examples=25)


if __name__ == "__main__":
    unittest.main(argv=[''])
