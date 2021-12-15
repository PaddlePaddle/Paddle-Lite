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
import random
import numpy as np


class TestBoxClipOp(AutoScanTest):
    def __init__(self, *args, **kwargs):
        AutoScanTest.__init__(self, *args, **kwargs)
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
        lod_data = [[1, 2, 3]]
        dim0 = draw(st.integers(min_value=4, max_value=10))
        dim1 = draw(st.integers(min_value=4, max_value=10))
        in_shape = [dim0, dim1, 4]
        iminfo_shape = [dim0, 3]

        def generate_input(*args, **kwargs):
            return np.random.random(in_shape).astype(np.float32)

        def generate_iminfo(*args, **kwargs):
            return np.random.random(iminfo_shape).astype(np.float32)

        box_clip_op = OpConfig(
            type="box_clip",
            inputs={"Input": ["input_data"],
                    "ImInfo": ["iminfo_data"]},
            outputs={"Output": ["output_data"]},
            attrs={})

        program_config = ProgramConfig(
            ops=[box_clip_op],
            weights={},
            inputs={
                "input_data": TensorConfig(
                    data_gen=partial(generate_input), lod=lod_data),
                "iminfo_data": TensorConfig(
                    data_gen=partial(generate_iminfo), lod=lod_data),
            },
            outputs=["output_data"])
        return program_config

    def sample_predictor_configs(self):
        return self.get_predictor_configs(), ["box_clip"], (1e-5, 1e-5)

    def add_ignore_pass_case(self):
        pass

    def test(self, *args, **kwargs):
        self.run_and_statis(quant=False, max_examples=25)


if __name__ == "__main__":
    unittest.main(argv=[''])
