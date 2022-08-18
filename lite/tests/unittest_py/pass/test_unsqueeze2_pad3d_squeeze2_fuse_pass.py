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


class TestUnsqueeze2Pad3dSqueeze2Fuse(FusePassAutoScanTest):
    def __init__(self, *args, **kwargs):
        FusePassAutoScanTest.__init__(self, *args, **kwargs)
        #x86
        self.enable_testing_on_place(
            TargetType.X86,
            PrecisionType.FP32,
            DataLayoutType.NCHW,
            thread=[1, 4])

    def is_program_valid(self,
                         program_config: ProgramConfig,
                         predictor_config: CxxConfig) -> bool:
        return True

    def sample_program_configs(self, draw):
        int32_values_1 = draw(st.integers(min_value=1, max_value=40))
        int32_values_2 = draw(st.integers(min_value=1, max_value=40))
        int32_values_3 = draw(st.integers(min_value=1, max_value=40))
        has_xshape = draw(st.booleans())
        unsqueeze2_input_shape = [int32_values_1, int32_values_2, 1, 1]

        if (has_xshape):
            unsqueeze2_op = OpConfig(
                type="unsqueeze2",
                inputs={"X": ["unsqueeze2_input_x"]},
                outputs={
                    "Out": ["unsqueeze2_out_data"],
                    "XShape": ["un_XShape_data"]
                },
                attrs={"axes": [3]})

            squeeze2_op = OpConfig(
                type="squeeze2",
                inputs={"X": ["p3d_output_data"]},
                outputs={
                    "Out": ["squeeze2_output"],
                    "XShape": ["squeeze2_output_XShape"]
                },
                attrs={
                    "axes": [3]  #required in pass
                })
        else:
            unsqueeze2_op = OpConfig(
                type="unsqueeze2",
                inputs={"X": ["unsqueeze2_input_x"]},
                outputs={"Out": ["unsqueeze2_out_data"]},
                attrs={"axes": [3]})

            squeeze2_op = OpConfig(
                type="squeeze2",
                inputs={"X": ["p3d_output_data"]},
                outputs={"Out": ["squeeze2_output"]},
                attrs={
                    "axes": [3]  #required in pass
                })

        pad3d_op = OpConfig(
            type="pad3d",
            inputs={"X": ["unsqueeze2_out_data"], },
            outputs={"Out": ["p3d_output_data"], },
            attrs={
                "paddings": [0, 0, 0, 0, 0, 0],
                "mode": "constant",
                "pad_value": 1.0,
                "data_format": "NCDHW"
            })

        ops = [unsqueeze2_op, pad3d_op, squeeze2_op]
        program_config = ProgramConfig(
            ops=ops,
            weights={},
            inputs={
                "unsqueeze2_input_x":
                TensorConfig(shape=unsqueeze2_input_shape),
            },
            outputs=["squeeze2_output"])
        return program_config

    def sample_predictor_configs(self):
        atol, rtol = 1e-5, 1e-5
        config_lists = self.get_predictor_configs()
        for config in config_lists:
            if config.target() in [TargetType.Metal]:
                atol, rtol = 1e-2, 1e-2

        return self.get_predictor_configs(), ["pad2d"], (atol, rtol)

    def add_ignore_pass_case(self):
        pass

    def test(self, *args, **kwargs):
        target_str = self.get_target()
        max_examples = 100
        self.run_and_statis(
            quant=False,
            max_examples=max_examples,
            passes=["lite_unsqueeze2_pad3d_squeeze2_fuse"])


if __name__ == "__main__":
    unittest.main(argv=[''])
