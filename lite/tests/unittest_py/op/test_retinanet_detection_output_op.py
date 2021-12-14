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

import numpy as np
from functools import partial

class TestRetinanetDetectionOutputOp(AutoScanTest):
    def __init__(self, *args, **kwargs):
        AutoScanTest.__init__(self, *args, **kwargs)
        self.enable_testing_on_place(TargetType.Host, PrecisionType.FP32, DataLayoutType.NCHW, thread=[1,2])
        self.enable_testing_on_place(TargetType.X86, PrecisionType.FP32, DataLayoutType.NCHW, thread=[1,2])

    def is_program_valid(self, program_config: ProgramConfig , predictor_config: CxxConfig) -> bool:
        return True

    def sample_program_configs(self, draw):
        in_shape = draw(st.lists(st.integers(min_value=2, max_value=10), min_size=2, max_size=4))
        N = draw(st.integers(min_value=1, max_value=10))
        C = draw(st.integers(min_value=3, max_value=5))
        fpn_num = draw(st.integers(min_value=3, max_value=5))
        anchor_num = [np.random.randint(1, 5) for i in range(fpn_num)]
        # draw(st.intergers(num_value=1, max_value=5)) 
        score_threshold = draw(st.sampled_from([0.05, 0.025]))
        nms_top_k = draw(st.sampled_from([1000, 500]))
        keep_top_k = draw(st.sampled_from([100, 50]))
        nms_threshold = draw(st.sampled_from([0.45, 0.3]))

        im_shape = [N, 3]
        # assume(len(in_shape) == 4)
        # assume(in_shape[1] % 2 == 0)

        def generate_bbox(*args, **kwargs):
            input = []
            for i in range(0, fpn_num):
                random_data = np.random.random([N, anchor_num[i], 4]).astype(np.float32)
                print(random_data.shape)
                # .astype(np.float32)
                input.append(random_data)
            print(input.shape)
            input = np.array(input).astype(np.float32)
            return input
    
        def generate_score(*args, **kwargs):
            input = []
            for i in range(0, fpn_num):
                random_data = np.random.random([N, anchor_num[i], C]).astype(np.float32)
                input.append(random_data)
            input = np.array(input).astype(np.float32)
            return input
    
        def generate_anchor(*args, **kwargs):
            input = []
            for i in range(0, fpn_num):
                random_data = np.random.random(1, 5, [4]).astype(np.float32)
                input.append(random_data)
            input = np.array(input).astype(np.float32)
            return input
    
        def generate_im(*args, **kwargs):
            input = []
            data = np.ones(kwargs['tensor_shape']).astype(np.float32)
            for i in range(0, fpn_num):
                data[i][0] = 40
                data[i][1] = 20
                data[i][2] = 0.6
            return data
    
        build_op = OpConfig(
            type = "retinanet_detection_output",
            inputs = {
                "BBoxes" : ["bboxes"],
                "Scores" : ["scores"],
                "Anchors" : ["anchors"],
                "ImInfo" : ["im_data"],
            },
            outputs = {"Out": ["output_data"]},
            attrs = {
                "score_threshold": score_threshold,
                "nms_top_k": nms_top_k,
                "nms_threshold": nms_threshold,
                "keep_top_k": keep_top_k,
                "nms_eta":"1.",
            })
        program_config = ProgramConfig(
            ops=[build_op],
            weights={},
            inputs={
                "BBoxes":
                TensorConfig(data_gen=partial(generate_bbox)),
                "Scores":
                TensorConfig(data_gen=partial(generate_score)),
                "Anchors":
                TensorConfig(data_gen=partial(generate_anchor)),
                "ImInfo":
                TensorConfig(data_gen=partial(generate_im, tensor_shape=im_shape)),
            },
            outputs=["output_data"])
        return program_config

    def sample_predictor_configs(self):
        return self.get_predictor_configs(), ["retinanet_detection_output"], (1e-5, 1e-5)

    def add_ignore_pass_case(self):
        pass

    def test(self, *args, **kwargs):
        self.run_and_statis(quant=False, max_examples=25)

if __name__ == "__main__":
    unittest.main(argv=[''])
