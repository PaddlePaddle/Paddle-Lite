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


class TestGenerateProposalsOp(AutoScanTest):
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
        in_shape = draw(
            st.lists(
                st.integers(
                    min_value=16, max_value=32),
                min_size=4,
                max_size=4))
        in_shape[0] = 1
        anchor_sizes = draw(
            st.sampled_from([[32.0], [32.0, 64.0], [64.0, 128.0],
                             [32.0, 64.0, 128.0]]))
        aspect_ratios = draw(
            st.sampled_from([[1.0], [1.0, 2.0], [0.5, 1.0, 2.0]]))
        variances = draw(
            st.lists(
                st.floats(
                    min_value=0.5, max_value=1.5),
                min_size=4,
                max_size=4))
        stride = draw(
            st.sampled_from([[16.0, 16.0], [24.0, 24.0], [16.0, 24.0]]))
        num_anchors = len(anchor_sizes) * len(aspect_ratios)

        anchor_generator_op = OpConfig(
            type="anchor_generator",
            inputs={"Input": ["input_data"]},
            outputs={
                "Anchors": ["anchors_data"],
                "Variances": ["variance_data"]
            },
            attrs={
                "anchor_sizes": anchor_sizes,
                "aspect_ratios": aspect_ratios,
                "stride": stride,
                "variances": variances,
                "offset": 0.5
            })

        scale = draw(st.floats(min_value=1, max_value=1))
        scores_shape = [in_shape[0], num_anchors, in_shape[2], in_shape[3]]
        bbox_delta_shape = [
            scores_shape[0], scores_shape[1] * 4, scores_shape[2],
            scores_shape[3]
        ]

        pre_nms_topN = draw(st.integers(min_value=2000, max_value=8000))
        post_nms_topN = draw(st.integers(min_value=1000, max_value=1500))
        nms_thresh = draw(st.floats(min_value=0.5, max_value=0.8))
        min_size = draw(st.floats(min_value=2, max_value=4))
        eta = draw(st.floats(min_value=0.5, max_value=1.5))

        def generate_im_info(*args, **kwargs):
            return np.array(
                [in_shape[2] * stride[0], in_shape[3] * stride[1],
                 scale]).astype(np.float32)

        generate_proposals_op = OpConfig(
            type="generate_proposals",
            inputs={
                "Scores": ["scores_data"],
                "BboxDeltas": ["bbox_delta_data"],
                "ImInfo": ["im_info_data"],
                "Anchors": ["anchors_data"],
                "Variances": ["variance_data"]
            },
            outputs={
                "RpnRois": ["rpn_rois_data"],
                "RpnRoiProbs": ["rpn_rois_probs_data"],
                "RpnRoisNum": ["rpn_rois_num_data"]
            },
            attrs={
                "pre_nms_topN": pre_nms_topN,
                "post_nms_topN": post_nms_topN,
                "nms_thresh": nms_thresh,
                "min_size": min_size,
                "eta": eta
            })
        program_config = ProgramConfig(
            ops=[anchor_generator_op, generate_proposals_op],
            weights={},
            inputs={
                "input_data": TensorConfig(shape=in_shape),
                "scores_data": TensorConfig(shape=scores_shape),
                "bbox_delta_data": TensorConfig(shape=bbox_delta_shape),
                "im_info_data":
                TensorConfig(data_gen=partial(generate_im_info))
            },
            outputs=[
                "rpn_rois_data", "rpn_rois_probs_data", "rpn_rois_num_data"
            ])
        return program_config

    def sample_predictor_configs(self):
        return self.get_predictor_configs(), ["anchor_generator"], (1e-5, 1e-5)

    def add_ignore_pass_case(self):
        pass

    def test(self, *args, **kwargs):
        self.run_and_statis(quant=False, max_examples=25)


if __name__ == "__main__":
    unittest.main(argv=[''])
