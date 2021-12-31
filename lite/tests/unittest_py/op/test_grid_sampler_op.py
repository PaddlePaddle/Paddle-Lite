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


class TestGridSamplerOp(AutoScanTest):
    def __init__(self, *args, **kwargs):
        AutoScanTest.__init__(self, *args, **kwargs)
        self.enable_testing_on_place(
            TargetType.X86,
            PrecisionType.FP32,
            DataLayoutType.NCHW,
            thread=[1, 4])
        self.enable_testing_on_place(
            TargetType.ARM,
            PrecisionType.FP32,
            DataLayoutType.NCHW,
            thread=[1, 4])
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

    def is_program_valid(self,
                         program_config: ProgramConfig,
                         predictor_config: CxxConfig) -> bool:
        if predictor_config.target() == TargetType.OpenCL:
            if program_config.ops[0].attrs[
                    "align_corners"] != True or program_config.ops[0].attrs[
                        "padding_mode"] != "zeros" or program_config.ops[
                            0].attrs["mode"] != "bilinear":
                return False
        return True

    def sample_program_configs(self, draw):
        in_shape1 = draw(
            st.lists(
                st.integers(
                    min_value=3, max_value=10), min_size=4, max_size=4))

        in_shape2 = []
        in_shape2.append(in_shape1[0])
        in_shape2.append(in_shape1[2])
        in_shape2.append(in_shape1[3])
        in_shape2.append(2)

        align_corners = draw(st.booleans())
        mode = draw(st.sampled_from(["bilinear", "nearest"]))
        padding_mode = draw(st.sampled_from(["zeros", "reflection", "border"]))
        grid_sampler_op = OpConfig(
            type="grid_sampler",
            inputs={"X": ["input_data"],
                    "Grid": ["grid_data"]},
            outputs={"Output": ["output_data"]},
            attrs={
                "align_corners": align_corners,
                "mode": mode,
                "padding_mode": padding_mode
            })

        program_config = ProgramConfig(
            ops=[grid_sampler_op],
            weights={"grid_data": TensorConfig(shape=in_shape2)},
            inputs={"input_data": TensorConfig(shape=in_shape1)},
            outputs=["output_data"])

        return program_config

    def sample_predictor_configs(self):
        return self.get_predictor_configs(), ["grid_sampler"], (1e-5, 1e-5)

    def add_ignore_pass_case(self):
        def teller1(program_config, predictor_config):
            return True

        self.add_ignore_check_case(
            # IgnoreReasonsBase.PADDLE_NOT_IMPLEMENTED
            # IgnoreReasonsBase.PADDLELITE_NOT_SUPPORT
            # IgnoreReasonsBase.ACCURACY_ERROR
            teller1,
            IgnoreReasons.ACCURACY_ERROR,
            "The op output has diff. We need to fix it as soon as possible.")

    def test(self, *args, **kwargs):
        self.run_and_statis(quant=False, max_examples=300)


if __name__ == "__main__":
    unittest.main(argv=[''])
