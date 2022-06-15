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


class TestPriorBoxOp(AutoScanTest):
    def __init__(self, *args, **kwargs):
        AutoScanTest.__init__(self, *args, **kwargs)
        self.enable_testing_on_place(
            TargetType.Host,
            PrecisionType.FP32,
            DataLayoutType.NCHW,
            thread=[1, 2])
        self.enable_testing_on_place(TargetType.NNAdapter, PrecisionType.FP32)
        self.enable_devices_on_nnadapter(
            device_names=["nvidia_tensorrt", "intel_openvino"])

    def is_program_valid(self,
                         program_config: ProgramConfig,
                         predictor_config: CxxConfig) -> bool:
        return True

    def sample_program_configs(self, draw):
        min_sizes = [2.0, 4.0]
        max_sizes = [5.0, 10.0]
        aspect_ratios = [2.0, 3.0]
        variances = [0.1, 0.1, 0.2, 0.2]
        flip = True
        clip = True
        layer_w = draw(st.integers(min_value=30, max_value=40))
        layer_h = draw(st.integers(min_value=30, max_value=40))
        image_w = draw(st.integers(min_value=40, max_value=50))
        image_h = draw(st.integers(min_value=40, max_value=50))

        step_w = float(image_w) / float(layer_w)
        step_h = float(image_h) / float(layer_h)

        input_channels = 2
        image_channels = 3
        batch_size = 10

        offset = 0.5
        min_max_aspect_ratios_order = draw(st.sampled_from([True, False]))
        if "intel_openvino" in self.get_nnadapter_device_name():
            assume(step_w == step_h)

        def generate_input(*args, **kwargs):
            return np.random.random((batch_size, image_channels, image_w,
                                     image_h)).astype('float32')

        def generate_image(*args, **kwargs):
            return np.random.random((batch_size, input_channels, layer_w,
                                     layer_h)).astype('float32')

        ops_config = OpConfig(
            type="prior_box",
            inputs={"Input": ["intput_data"],
                    "Image": ["intput_image"]},
            outputs={"Boxes": ["output_boxes"],
                     "Variances": ["variances"]},
            attrs={
                "min_sizes": min_sizes,
                "max_sizes": max_sizes,
                "aspect_ratios": aspect_ratios,
                "variances": variances,
                "flip": flip,
                "clip": clip,
                "step_w": step_w,
                "step_h": step_h,
                "offset": offset,
                "min_max_aspect_ratios_order": min_max_aspect_ratios_order,
            }, )

        program_config = ProgramConfig(
            ops=[ops_config],
            weights={},
            inputs={
                "intput_data": TensorConfig(data_gen=partial(generate_input)),
                "intput_image": TensorConfig(data_gen=partial(generate_image)),
            },
            outputs=["output_boxes", "variances"])

        return program_config

    def sample_predictor_configs(self):
        return self.get_predictor_configs(), ["prior_box"], (1e-5, 1e-5)

    def add_ignore_pass_case(self):
        pass

    def test(self, *args, **kwargs):
        self.run_and_statis(quant=False, max_examples=25)


if __name__ == "__main__":
    unittest.main(argv=[''])
