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
from functools import partial
import hypothesis
from hypothesis import given, settings, seed, example, assume
import hypothesis.strategies as st
import numpy as np


class TestSequenceToplAvgPoolingOp(AutoScanTest):
    def __init__(self, *args, **kwargs):
        AutoScanTest.__init__(self, *args, **kwargs)
        self.enable_testing_on_place(
            TargetType.X86, [PrecisionType.FP32],
            DataLayoutType.NCHW,
            thread=[1, 4])

    def is_program_valid(self,
                         program_config: ProgramConfig,
                         predictor_config: CxxConfig) -> bool:
        return True

    def sample_program_configs(self, draw):
        topks = draw(st.sampled_from([[1, 3], [1, 3, 5], [1, 3, 5, 7]]))
        channel_num = draw(st.sampled_from([1, 3, 5, 7]))
        dim = draw(st.sampled_from([10, 12, 64]))
        row = draw(st.sampled_from([[0, 30], [0, 40], [0, 50]]))
        col = draw(st.sampled_from([[0, 25], [0, 35], [0, 45]]))
        feature = [row[i] * col[i] for i in range(len(row))]
        lod_ = [[x * channel_num for x in feature]]

        def generate_input(*args, **kwargs):
            return np.arange(sum(feature) * channel_num).astype(np.float32)

        def generate_row(*args, **kwargs):
            return np.random.random((sum(row), dim)).astype(np.float32)

        def generate_column(*args, **kwargs):
            return np.random.random((sum(col), dim)).astype(np.float32)

        ops_config = OpConfig(
            type="sequence_topk_avg_pooling",
            inputs={
                "X": ["input_data"],
                "ROW": ["row"],
                "COLUMN": ["column"]
            },
            outputs={"Out": ["output_data"],
                     "pos": ["pos"]},
            attrs={"topks": topks,
                   "channel_num": channel_num})

        program_config = ProgramConfig(
            ops=[ops_config],
            weights={},
            inputs={
                "input_data": TensorConfig(
                    data_gen=partial(generate_input), lod=lod_),
                "row": TensorConfig(
                    data_gen=partial(generate_row), lod=[row]),
                "column": TensorConfig(
                    data_gen=partial(generate_column), lod=[col])
            },
            outputs=["output_data", "pos"])

        return program_config

    def sample_predictor_configs(self):
        return self.get_predictor_configs(), ["sequence_topk_avg_pooling"], (
            1e-5, 1e-5)

    def add_ignore_pass_case(self):
        pass

    def test(self, *args, **kwargs):
        self.run_and_statis(quant=False, max_examples=200)


if __name__ == "__main__":
    unittest.main(argv=[''])
