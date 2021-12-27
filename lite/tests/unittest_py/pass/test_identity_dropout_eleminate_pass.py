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

from auto_scan_test import FusePassAutoScanTest, IgnoreReasons
from program_config import TensorConfig, ProgramConfig, OpConfig, CxxConfig, TargetType, PrecisionType, DataLayoutType, Place
import numpy as np
from functools import partial
from typing import Optional, List, Callable, Dict, Any, Set
import unittest

import hypothesis
from hypothesis import given, settings, seed, example, assume, reproduce_failure
import hypothesis.strategies as st


class TestIdentifyDropoutEleminateFuse(FusePassAutoScanTest):
    def __init__(self, *args, **kwargs):
        FusePassAutoScanTest.__init__(self, *args, **kwargs)
        self.enable_testing_on_place(
            TargetType.ARM, [PrecisionType.FP32],
            DataLayoutType.NCHW,
            thread=[1, 4])
        self.enable_testing_on_place(
            TargetType.X86, [PrecisionType.FP32],
            DataLayoutType.NCHW,
            thread=[1, 4])

    def is_program_valid(self,
                         program_config: ProgramConfig,
                         predictor_config: CxxConfig) -> bool:
        return True

    def sample_program_configs(self, draw):
        in_shape = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=8), min_size=2, max_size=4))
        dropout_prob_data = draw(st.floats(min_value=0.0, max_value=1.0))
        seed_data = draw(st.integers(min_value=0.0, max_value=1.0))
        fix_seed = draw(st.booleans())

        threshold = draw(st.floats(min_value=0, max_value=1))
        scale = draw(st.floats(min_value=0.5, max_value=5))
        offset = draw(st.floats(min_value=0, max_value=1))

        hard_swish_op = OpConfig(
            type="hard_swish",
            inputs={"X": ["input_data"]},
            outputs={"Out": ["hard_swish_output_data"]},
            attrs={"threshold": threshold,
                   "scale": scale,
                   "offset": offset})

        dropout_op = OpConfig(
            type="dropout",
            inputs={"X": ["hard_swish_output_data"]},
            outputs={"Out": ["output_data"],
                     "Mask": ["output_data_mask"]},
            attrs={
                "dropout_implementation": "upscale_in_train",
                "is_test": True,
                "dropout_prob": dropout_prob_data,
                "fix_seed": fix_seed,
                "seed": seed_data
            })

        ops = [hard_swish_op, dropout_op]
        program_config = ProgramConfig(
            ops=ops,
            weights={},
            inputs={"input_data": TensorConfig(shape=in_shape)},
            outputs=["output_data"])
        return program_config

    def sample_predictor_configs(self):
        config = CxxConfig()
        return self.get_predictor_configs(), ['hard_swish'], (1e-5, 1e-5)

    def add_ignore_pass_case(self):
        pass

    def test(self, *args, **kwargs):
        self.run_and_statis(
            quant=False,
            max_examples=25,
            passes=["lite_identify_dropout_eliminate_pass"])


if __name__ == "__main__":
    unittest.main(argv=[''])
