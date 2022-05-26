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
import numpy as np
import hypothesis
from hypothesis import given, settings, seed, example, assume
import hypothesis.strategies as st
import argparse


class TestRoiAlignOp(AutoScanTest):
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
                    min_value=4, max_value=64), min_size=4, max_size=4))
        spatial_scale = draw(st.floats(min_value=0.1, max_value=1.0))
        pooled_height = draw(st.integers(min_value=1, max_value=4))
        pooled_width = draw(st.integers(min_value=1, max_value=4))
        sampling_ratio = draw(st.sampled_from([-1, 4, 8]))
        aligned = draw(st.booleans())
        roi_num_data = np.random.randint(
            low=0, high=4, size=[in_shape[0]]).astype(np.int32)
        num_rois = np.sum(roi_num_data)
        case_type = draw(st.sampled_from(
            ["c1", "c2"]))  #c1 has 2 inputs, c2 has 3 inputs

        def generate_roisnum(*args, **kwargs):
            return roi_num_data

        def generate_input(*args, **kwargs):
            return np.random.random(in_shape).astype(np.float32)

        def generate_lod():
            lod_num = []
            lod_num.append(0)
            for i in range(in_shape[0] - 1):
                lod_num.append(i + 1)
            lod_num.append(num_rois)
            return lod_num

        height = in_shape[2]
        width = in_shape[3]
        x1 = draw(
            st.integers(
                min_value=0, max_value=width // spatial_scale - pooled_width))
        y1 = draw(
            st.integers(
                min_value=0, max_value=height // spatial_scale -
                pooled_height))
        x2 = draw(
            st.integers(
                min_value=x1 + pooled_width, max_value=width // spatial_scale))
        y2 = draw(
            st.integers(
                min_value=y1 + pooled_height,
                max_value=height // spatial_scale))

        def generate_rois(*args, **kwargs):
            a = np.array([x1, y1, x2, y2]).astype(np.float32).reshape([1, 4])
            b = a.repeat(num_rois, axis=0)
            b.reshape([num_rois, 4])
            return b

        if case_type == "c2":
            roi_align_op = OpConfig(
                type="roi_align",
                inputs={
                    "X": ["input_data"],
                    "ROIs": ["rois_data"],
                    "RoisNum": ["roisnum_data"]
                },
                outputs={"Out": ["output_data"]},
                attrs={
                    "spatial_scale": spatial_scale,
                    "pooled_height": pooled_height,
                    "pooled_width": pooled_width,
                    "sampling_ratio": sampling_ratio,
                    "aligned": aligned
                })
            program_config = ProgramConfig(
                ops=[roi_align_op],
                weights={},
                inputs={
                    "input_data":
                    TensorConfig(data_gen=partial(generate_input)),
                    "rois_data": TensorConfig(data_gen=partial(generate_rois)),
                    "roisnum_data":
                    TensorConfig(data_gen=partial(generate_roisnum))
                },
                outputs=["output_data"])
        else:
            roi_align_op = OpConfig(
                type="roi_align",
                inputs={"X": ["input_data"],
                        "ROIs": ["rois_data"]},
                outputs={"Out": ["output_data"]},
                attrs={
                    "spatial_scale": spatial_scale,
                    "pooled_height": pooled_height,
                    "pooled_width": pooled_width,
                    "sampling_ratio": sampling_ratio,
                    "aligned": aligned
                })
            program_config = ProgramConfig(
                ops=[roi_align_op],
                weights={},
                inputs={
                    "input_data":
                    TensorConfig(data_gen=partial(generate_input)),
                    "rois_data": TensorConfig(
                        data_gen=partial(generate_rois), lod=[generate_lod()])
                },
                outputs=["output_data"])
        return program_config

    def sample_predictor_configs(self):
        return self.get_predictor_configs(), ["roi_align"], (1e-5, 1e-5)

    def add_ignore_pass_case(self):
        pass

    def test(self, *args, **kwargs):
        self.run_and_statis(quant=False, max_examples=200)


if __name__ == "__main__":
    unittest.main(argv=[''])
