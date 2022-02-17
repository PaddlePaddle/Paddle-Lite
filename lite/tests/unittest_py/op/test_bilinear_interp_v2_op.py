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


class TestBilinearV2Op(AutoScanTest):
    def __init__(self, *args, **kwargs):
        AutoScanTest.__init__(self, *args, **kwargs)
        self.enable_testing_on_place(
            TargetType.ARM,
            PrecisionType.FP32,
            DataLayoutType.NCHW,
            thread=[1, 4])
        # x86 has diff
        self.enable_testing_on_place(
            TargetType.X86,
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
        metal_places = [
            Place(TargetType.Metal, PrecisionType.FP32,
                  DataLayoutType.MetalTexture2DArray),
            Place(TargetType.Metal, PrecisionType.FP16,
                  DataLayoutType.MetalTexture2DArray),
            Place(TargetType.ARM, PrecisionType.FP32),
            Place(TargetType.Host, PrecisionType.FP32)
        ]
        self.enable_testing_on_place(places=metal_places)
        self.enable_testing_on_place(TargetType.NNAdapter, PrecisionType.FP32)
        self.enable_devices_on_nnadapter(device_names=["cambricon_mlu"])

    def is_program_valid(self,
                         program_config: ProgramConfig,
                         predictor_config: CxxConfig) -> bool:
        return True

    def sample_program_configs(self, draw):
        batch = draw(st.integers(min_value=1, max_value=4))
        channel = draw(st.integers(min_value=1, max_value=32))
        height = draw(st.integers(min_value=3, max_value=100))
        width = draw(st.integers(min_value=3, max_value=100))
        in_shape = [batch, channel, height, width]
        out_size_shape = draw(st.sampled_from([[1, 2]]))
        align_corners = draw(st.booleans())
        align_mode = draw(st.sampled_from([0, 1]))
        out_h = draw(st.integers(min_value=3, max_value=10))
        out_w = draw(st.integers(min_value=3, max_value=10))
        scale = draw(
            st.lists(
                st.floats(
                    min_value=0.1, max_value=0.9),
                min_size=2,
                max_size=2))

        def generate_input(*args, **kwargs):
            return np.random.random(in_shape).astype(np.float32)

        def generate_out_size(*args, **kwargs):
            return np.random.randint(1, 100, size=out_size_shape)

        def generate_size_tensor(*args, **kwargs):
            return np.random.randint(3, 100, [1]).astype(np.int32)

        def generate_scale(*args, **kwargs):
            return np.random.random([1]).astype(np.int32)

        bilinear_interp_v2_op = OpConfig(
            type="bilinear_interp_v2",
            inputs={
                "X": ["input_data"],
                "OutSize": ["out_size_data"],
                "SizeTensor": ["size_tensor_data1", "size_tensor_data2"],
                "Scale": ["scale_data"]
            },
            outputs={"Out": ["output_data"]},
            attrs={
                "data_layout": "NCHW",
                "out_d": 0,
                "out_h": out_h,
                "out_w": out_w,
                "scale": scale,
                "interp_method": "bilinear",
                "align_corners": align_corners,
                "align_mode": align_mode
            })
        program_config = ProgramConfig(
            ops=[bilinear_interp_v2_op],
            weights={},
            inputs={
                "input_data": TensorConfig(data_gen=partial(generate_input)),
                "out_size_data":
                TensorConfig(data_gen=partial(generate_out_size)),
                "size_tensor_data1":
                TensorConfig(data_gen=partial(generate_size_tensor)),
                "size_tensor_data2":
                TensorConfig(data_gen=partial(generate_size_tensor)),
                "scale_data": TensorConfig(data_gen=partial(generate_scale))
            },
            outputs=["output_data"])
        return program_config

    def sample_predictor_configs(self):
        atol, rtol = 1e-4, 1e-4
        target_str = self.get_target()
        if target_str == "Metal":
            atol, rtol = 3e-1, 3e-1
        return self.get_predictor_configs(), ["bilinear_interp_v2"], (atol,
                                                                      rtol)

    def add_ignore_pass_case(self):
        pass

    def test(self, *args, **kwargs):
        self.run_and_statis(quant=False, max_examples=100)


if __name__ == "__main__":
    unittest.main(argv=[''])
