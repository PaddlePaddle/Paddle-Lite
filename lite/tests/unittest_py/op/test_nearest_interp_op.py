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


class TestNearestInterpOp(AutoScanTest):
    def __init__(self, *args, **kwargs):
        AutoScanTest.__init__(self, *args, **kwargs)
        # precision bugs will be fix in the future
        #self.enable_testing_on_place(
        #    TargetType.ARM, [PrecisionType.FP16, PrecisionType.FP32],
        #    DataLayoutType.NCHW,
        #    thread=[1, 4])
        #self.enable_testing_on_place(
        #    TargetType.X86,
        #    PrecisionType.FP32,
        #    DataLayoutType.NCHW,
        #    thread=[1, 4])
        #self.enable_testing_on_place(
        #    TargetType.Metal,
        #    PrecisionType.FP32,
        #    DataLayoutType.NCHW,
        #    thread=[1, 4])
        #opencl_places = [
        #    Place(TargetType.OpenCL, PrecisionType.FP16,
        #          DataLayoutType.ImageDefault), Place(
        #              TargetType.OpenCL, PrecisionType.FP16,
        #              DataLayoutType.ImageFolder),
        #    Place(TargetType.OpenCL, PrecisionType.FP32, DataLayoutType.NCHW),
        #    Place(TargetType.OpenCL, PrecisionType.Any,
        #          DataLayoutType.ImageDefault), Place(
        #              TargetType.OpenCL, PrecisionType.Any,
        #              DataLayoutType.ImageFolder),
        #    Place(TargetType.OpenCL, PrecisionType.Any, DataLayoutType.NCHW),
        #    Place(TargetType.Host, PrecisionType.FP32)
        #]
        #self.enable_testing_on_place(places=opencl_places)

    def is_program_valid(self,
                         program_config: ProgramConfig,
                         predictor_config: CxxConfig) -> bool:
        in_shape = list(program_config.inputs["input_data_x"].shape)
        scale_data = program_config.inputs["Scale"].data
        SizeTensor = list(program_config.inputs["SizeTensor"].shape)
        # paddle not support fp16
        if predictor_config.precision() == PrecisionType.FP16:
            return False
        if in_shape[2] * scale_data[0] < 1 or in_shape[3] * scale_data[0] < 1:
            return False
        return True

    def sample_program_configs(self, draw):
        in_num = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=4), min_size=1, max_size=1))
        in_c_h_w = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=128),
                min_size=3,
                max_size=3))
        X_shape = in_num + in_c_h_w
        Scale_shape = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=64), min_size=1, max_size=1))
        Tensor_shape = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=32), min_size=4, max_size=4))
        align_corners = draw(st.booleans())
        scale = draw(st.floats(min_value=0.1, max_value=10.0))
        interp_method = draw(st.sampled_from(["nearest"]))
        out_w = draw(st.integers(min_value=1, max_value=32))
        out_h = draw(st.integers(min_value=1, max_value=32))
        data_layout = draw(st.sampled_from(["NCHW"]))

        def generate_input1(*args, **kwargs):
            return np.random.normal(0.0, 1.0, X_shape).astype(np.float32)

        def generate_input1_fp16(*args, **kwargs):
            return np.random.normal(0.0, 1.0, X_shape).astype(np.float16)

        nearest_interp = OpConfig(
            type="nearest_interp",
            inputs={"X": ["input_data_x"],
                    "Scale": ["Scale"]},
            outputs={"Out": ["output_data"]},
            attrs={
                "data_layout": data_layout,
                "scale": scale,
                "out_w": out_w,
                "out_h": out_h,
                "interp_method": interp_method,
                "align_corners": align_corners
            })
        program_config = ProgramConfig(
            ops=[nearest_interp],
            weights={},
            inputs={
                "input_data_x": TensorConfig(data_gen=generate_input1),
                "SizeTensor": TensorConfig(shape=Tensor_shape),
                "Scale": TensorConfig(shape=Scale_shape)
            },
            outputs={"output_data"})
        return program_config

    def sample_predictor_configs(self):
        return self.get_predictor_configs(), ["nearest_interp"], (1e-5, 1e-5)

    def add_ignore_pass_case(self):
        pass

    def test(self, *args, **kwargs):
        self.run_and_statis(quant=False, max_examples=100)


if __name__ == "__main__":
    unittest.main(argv=[''])
