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
sys.path.append('..')

from auto_scan_test import FusePassAutoScanTest
from program_config import TensorConfig, ProgramConfig, OpConfig, CxxConfig, TargetType, PrecisionType, DataLayoutType, Place
import numpy as np
from functools import partial
from typing import Optional, List, Callable, Dict, Any, Set
import unittest

import hypothesis
from hypothesis import given, settings, seed, example, assume, reproduce_failure
import hypothesis.strategies as st


class TestScaleActiveFusePass(FusePassAutoScanTest):
    def __init__(self, *args, **kwargs):
        FusePassAutoScanTest.__init__(self, *args, **kwargs)
        self.enable_testing_on_place(
            TargetType.ARM,
            PrecisionType.FP32,
            DataLayoutType.NCHW,
            thread=[1, 4])

    def is_program_valid(self,
                         program_config: ProgramConfig,
                         predictor_config: CxxConfig) -> bool:
        #Output has diff
        if predictor_config.target() == TargetType.ARM:
            return False
        else:
            return True

    def sample_program_configs(self, draw):
        in_shape_x = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=20), min_size=4, max_size=4))
        threshold = draw(st.floats(min_value=0, max_value=1))
        alpha = draw(st.floats(min_value=0, max_value=1))
        scale = draw(st.floats(min_value=0.5, max_value=5))
        bias = draw(st.floats(min_value=0, max_value=1))
        bias_after_scale = draw(st.sampled_from([True]))

        act_type = draw(st.sampled_from(['relu', 'relu6', 'leaky_relu']))

        def generate_act_attrs(act_type_str):
            attrs = {}
            if act_type_str == 'relu6':
                attrs = {"threshold": threshold}
            elif act_type_str == 'leaky_relu':
                attrs = {"alpha": alpha}
            elif act_type_str == 'relu':
                attrs = {}
            return attrs

        scale_op = OpConfig(
            type="scale",
            inputs={"X": ["input_data_x"]},
            outputs={"Out": ["scale_output_data"]},
            attrs={
                "scale": scale,
                "bias": bias,
                "bias_after_scale": bias_after_scale
            })

        active_op = OpConfig(
            type=act_type,
            inputs={"X": ["scale_output_data"]},
            outputs={"Out": ["output_data"]},
            attrs=generate_act_attrs(act_type))

        ops = [scale_op, active_op]
        program_config = ProgramConfig(
            ops=ops,
            weights={},
            inputs={"input_data_x": TensorConfig(shape=in_shape_x)},
            outputs=["output_data"])
        return program_config

    def sample_predictor_configs(self):
        if self.get_target() == 'OpenCL':
            return self.get_predictor_configs(
            ), ['io_copy', 'layout', 'scale', 'layout', 'io_copy'], (1e-5,
                                                                     1e-5)
        else:
            return self.get_predictor_configs(), ['scale'], (1e-5, 1e-5)

    def add_ignore_pass_case(self):
        pass

    def test(self, *args, **kwargs):
        self.run_and_statis(
            quant=False,
            max_examples=25,
            passes=["lite_scale_activation_fuse_pass"])


if __name__ == "__main__":
    unittest.main(argv=[''])
