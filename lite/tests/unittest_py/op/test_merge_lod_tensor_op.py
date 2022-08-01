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


class TestMergeLodTensorOp(AutoScanTest):
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
        n = draw(st.integers(min_value=1, max_value=32))

        def generate_input_data_in_shape(*args, **kwargs):
            return np.random.random([n, 1]).astype('float32')

        def generate_mask_data_in_shape(*args, **kwargs):
            return np.expand_dims(
                np.random.randint(
                    low=0, high=1, size=[n]).astype('int32'),
                axis=1)

        def generate_in_true_data_in_shape(*args, **kwargs):
            return np.expand_dims(np.random.rand(n).astype('float32'), axis=1)

        def generate_in_false_data_in_shape(*args, **kwargs):
            return np.expand_dims(np.random.rand(n).astype('float32'), axis=1)

        cast_x = OpConfig(
            type="cast",
            inputs={"X": ["input_data_mask"], },
            outputs={"Out": ["cast_data_mask"], },
            attrs={"in_dtype": int(2),
                   "out_dtype": int(0)})
        cast_x.outputs_dtype = {"cast_data_mask": np.bool_}
        match_matrix_tensor_op = OpConfig(
            type="merge_lod_tensor",
            inputs={
                "X": ["input_data_x"],
                "Mask": ["cast_data_mask"],
                "InTrue": ["InTrue"],
                "InFalse": ["InFalse"]
            },
            outputs={"Out": ["output_data"]},
            attrs={"level": 0})
        program_config = ProgramConfig(
            ops=[cast_x, match_matrix_tensor_op],
            weights={},
            inputs={
                "input_data_x":
                TensorConfig(data_gen=generate_input_data_in_shape),
                "input_data_mask":
                TensorConfig(data_gen=generate_mask_data_in_shape),
                "InTrue":
                TensorConfig(data_gen=generate_in_true_data_in_shape),
                "InFalse":
                TensorConfig(data_gen=generate_in_false_data_in_shape)
            },
            outputs={"output_data"})
        return program_config

    def sample_predictor_configs(self):
        return self.get_predictor_configs(), ["merge_lod_tensor"], (1e-5, 1e-5)

    def add_ignore_pass_case(self):
        pass

    def test(self, *args, **kwargs):
        self.run_and_statis(quant=False, max_examples=25)


if __name__ == "__main__":
    unittest.main(argv=[''])
