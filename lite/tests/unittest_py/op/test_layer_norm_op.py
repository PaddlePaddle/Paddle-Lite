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


class TestLayerNormOp(AutoScanTest):
    def __init__(self, *args, **kwargs):
        AutoScanTest.__init__(self, *args, **kwargs)
        self.enable_testing_on_place(
            TargetType.X86,
            PrecisionType.FP32,
            DataLayoutType.NCHW,
            thread=[1, 2])
        self.enable_testing_on_place(
            TargetType.ARM,
            PrecisionType.FP32,
            DataLayoutType.NCHW,
            thread=[1, 2, 4])
        opencl_places = [
            Place(TargetType.OpenCL, PrecisionType.FP16,
                  DataLayoutType.ImageDefault), Place(
                      TargetType.OpenCL, PrecisionType.FP16,
                      DataLayoutType.ImageFolder),
            Place(TargetType.OpenCL, PrecisionType.FP32, DataLayoutType.NCHW),
            Place(TargetType.OpenCL, PrecisionType.Any,
                  DataLayoutType.ImageDefault), Place(
                      TargetType.OpenCL, PrecisionType.Any,
                      DataLayoutType.ImageFolder),
            Place(TargetType.OpenCL, PrecisionType.Any, DataLayoutType.NCHW),
            Place(TargetType.Host, PrecisionType.FP32)
        ]
        self.enable_testing_on_place(places=opencl_places)
        self.enable_testing_on_place(TargetType.NNAdapter, PrecisionType.FP32)
        self.enable_devices_on_nnadapter(device_names=["intel_openvino"])

    def is_program_valid(self,
                         program_config: ProgramConfig,
                         predictor_config: CxxConfig) -> bool:
        return True

    def sample_program_configs(self, draw):
        in_shape = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=64), min_size=2, max_size=3))
        in_shape.insert(0, draw(st.integers(min_value=1, max_value=64)))
        epsilon = draw(st.floats(min_value=0.0001, max_value=0.0005))
        begin_norm_axis = draw(st.sampled_from([1, 2, 3]))
        assume(begin_norm_axis < len(in_shape))

        def generate_input(*args, **kwargs):
            return np.random.normal(0.0, 1.0, in_shape).astype(np.float32)

        channel_dim = 1
        for dim in range(begin_norm_axis, len(in_shape)):
            channel_dim = channel_dim * in_shape[dim]

        def generate_scale(*args, **kwargs):
            return np.random.random([channel_dim]).astype(np.float32)

        def generate_bias(*args, **kwargs):
            return np.random.random([channel_dim]).astype(np.float32)

        def generate_inputs(target_nnadapter):
            inputs1 = {}
            inputs2 = {}
            if target_nnadapter:
                inputs1 = {"X": ["input_data"]}
                inputs2 = {
                    "input_data":
                    TensorConfig(data_gen=partial(generate_input)),
                }
            else:
                inputs1 = {
                    "X": ["input_data"],
                    "Scale": ["scale_data"],
                    "Bias": ["bias_data"]
                }
                inputs2 = {
                    "input_data":
                    TensorConfig(data_gen=partial(generate_input)),
                    "scale_data":
                    TensorConfig(data_gen=partial(generate_scale)),
                    "bias_data": TensorConfig(data_gen=partial(generate_bias))
                }
            return [inputs1, inputs2]

        def generate_attrs(target_nnadapter):
            attrs = {}
            if target_nnadapter:
                attrs = {
                    "epsilon": epsilon,
                    "Scale": generate_scale(),
                    "Bias": generate_bias(),
                    "begin_norm_axis": begin_norm_axis
                }
            else:
                attrs = {
                    "epsilon": epsilon,
                    "begin_norm_axis": begin_norm_axis
                }
            return attrs

        inputs = generate_inputs(self.get_target() == 'NNAdapter')
        attrs = generate_attrs(self.get_target() == 'NNAdapter')
        run_op = OpConfig(
            type="layer_norm",
            inputs=inputs[0],
            outputs={
                "Y": ["output_data"],
                "Mean": ["mean_data"],
                "Variance": ["variance_data"],
            },
            attrs=attrs)
        program_config = ProgramConfig(
            ops=[run_op],
            weights={},
            inputs=inputs[1],
            outputs=["output_data", "mean_data", "variance_data"])
        return program_config

    def sample_predictor_configs(self):
        return self.get_predictor_configs(), ["layer_norm"], (5e-5, 5e-5)

    def add_ignore_pass_case(self):
        pass

    def test(self, *args, **kwargs):
        self.run_and_statis(quant=False, max_examples=75)


if __name__ == "__main__":
    unittest.main(argv=[''])
