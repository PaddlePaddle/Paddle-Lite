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


class TestBeamSearchDecodeOp(AutoScanTest):
    def __init__(self, *args, **kwargs):
        AutoScanTest.__init__(self, *args, **kwargs)
        # it doesn't suppor std::vector<Tensor>
        # self.enable_testing_on_place(
        #     TargetType.Host,
        #     PrecisionType.FP32,
        #     DataLayoutType.NCHW,
        #     thread=[1, 4])

    def is_program_valid(self,
                         program_config: ProgramConfig,
                         predictor_config: CxxConfig) -> bool:
        # doesn't support tensorList type
        return False

    def sample_program_configs(self, draw):
        in_shape = draw(st.sampled_from([[5, 1]]))
        lod_data = draw(st.sampled_from([[[0, 1, 2], [0, 2, 4]]]))

        def generate_pre_ids(*args, **kwargs):
            return np.random.random(in_shape).astype(np.int64)

        def generate_pre_score(*args, **kwargs):
            return np.random.random(in_shape).astype(np.float32)

        beam_search_ops = OpConfig(
            type="beam_search_decode",
            inputs={
                "Ids": ["ids_data", "ids_data2"],
                "Scores": ["scores_data", "scores_data2"]
            },
            outputs={
                "SentenceIds": ["sentence_ids_data"],
                "SentenceScores": ["sentence_scores_data"]
            },
            attrs={"beam_size": in_shape[0],
                   "end_id": 0})
        program_config = ProgramConfig(
            ops=[beam_search_ops],
            weights={},
            inputs={
                "ids_data": TensorConfig(
                    data_gen=partial(generate_pre_ids), lod=lod_data),
                "ids_data2": TensorConfig(
                    data_gen=partial(generate_pre_ids), lod=lod_data),
                "scores_data": TensorConfig(
                    data_gen=partial(generate_pre_score), lod=lod_data),
                "scores_data2": TensorConfig(
                    data_gen=partial(generate_pre_score), lod=lod_data),
            },
            outputs=["sentence_ids_data", "sentence_scores_data"])
        return program_config

    def sample_predictor_configs(self):
        return self.get_predictor_configs(), ["beam_search_decode"], (1e-5,
                                                                      1e-5)

    def add_ignore_pass_case(self):
        pass

    def test(self, *args, **kwargs):
        self.run_and_statis(quant=False, max_examples=25)


if __name__ == "__main__":
    unittest.main(argv=[''])
