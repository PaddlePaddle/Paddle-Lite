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


class TestMatrixNMSOp(AutoScanTest):
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
        shape0 = draw(st.integers(min_value=1, max_value=64))
        shape1 = draw(st.sampled_from(
            [4]))  #PADDLE_ENFORCE_EQ(box_dims[2] == 4)
        shape2 = draw(st.integers(min_value=1, max_value=64))
        shape3 = draw(st.integers(min_value=1, max_value=64))
        X_shape = [shape0, shape2, shape1]  #[N, M, 4]
        Y_shape = [shape0, shape3, shape2]  #[N, C, M]

        keep_top_k = draw(st.sampled_from([1]))
        normalized = draw(st.booleans())
        background_label = draw(st.sampled_from([0, 1]))
        score_threshold = draw(st.floats(min_value=0.1, max_value=1.0))
        post_threshold = draw(st.floats(min_value=0.1, max_value=1.0))
        nms_top_k = draw(st.integers(min_value=1, max_value=1))
        keep_top_k = draw(st.integers(min_value=1, max_value=1))
        use_gaussian = draw(st.booleans())
        gaussian_sigma = draw(st.floats(min_value=0.1, max_value=1.0))

        matrix_nms_op = OpConfig(
            type="matrix_nms",
            inputs={
                "BBoxes": ["input_data_BBoxes"],
                "Scores": ["input_data_Scores"]
            },
            outputs={
                "Out": ["output_data"],
                "Index": ["output_index"],
                "RoisNum": ["RoisNum"]
            },
            attrs={
                "background_label": background_label,
                "score_threshold": score_threshold,
                "post_threshold": post_threshold,
                "nms_top_k": nms_top_k,
                "keep_top_k": keep_top_k,
                "normalized": normalized,
                "use_gaussian": use_gaussian,
                "gaussian_sigma": gaussian_sigma
            })

        program_config = ProgramConfig(
            ops=[matrix_nms_op],
            weights={},
            inputs={
                "input_data_BBoxes": TensorConfig(shape=X_shape),
                "input_data_Scores": TensorConfig(shape=Y_shape)
            },
            outputs={"output_data"})
        return program_config

    def sample_predictor_configs(self):
        return self.get_predictor_configs(), ["matrix_nms"], (1e-5, 1e-5)

    def add_ignore_pass_case(self):
        pass

    def test(self, *args, **kwargs):
        self.run_and_statis(quant=False, max_examples=50)


if __name__ == "__main__":
    unittest.main(argv=[''])
