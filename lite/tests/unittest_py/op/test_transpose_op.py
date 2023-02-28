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
from functools import partial
import random
import numpy as np


class TestTransposeOp(AutoScanTest):
    def __init__(self, *args, **kwargs):
        AutoScanTest.__init__(self, *args, **kwargs)
        x86_places = [
            Place(TargetType.X86, PrecisionType.FP32, DataLayoutType.NCHW)
        ]
        self.enable_testing_on_place(places=x86_places)

        arm_places = [
            Place(TargetType.ARM, PrecisionType.FP32, DataLayoutType.NCHW)
        ]
        self.enable_testing_on_place(places=arm_places)

        # opencl having diffs , big diff
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
        self.enable_testing_on_place(
            TargetType.ARM,
            PrecisionType.FP16,
            DataLayoutType.NCHW,
            thread=[1, 4])
        self.enable_testing_on_place(TargetType.NNAdapter, PrecisionType.FP32)
        self.enable_devices_on_nnadapter(device_names=[
            "kunlunxin_xtcl", "nvidia_tensorrt", "intel_openvino"
        ])

    def is_program_valid(self,
                         program_config: ProgramConfig,
                         predictor_config: CxxConfig) -> bool:
        return True

    def sample_program_configs(self, draw):
        N = draw(st.integers(min_value=1, max_value=4))
        C = draw(st.integers(min_value=1, max_value=128))
        H = draw(st.integers(min_value=1, max_value=128))
        W = draw(st.integers(min_value=1, max_value=128))
        in_shape = draw(st.sampled_from([[N, C, H, W]]))
        # tranpose only support float32
        # so we only feed input np.float
        in_dtype = draw(st.sampled_from([np.float32]))
        use_mkldnn_data = False
        target = self.get_target()
        if (target == "X86"):
            use_mkldnn_data = True

        axis_int32_data = draw(
            st.lists(
                st.integers(
                    min_value=0, max_value=3), min_size=3, max_size=4))
        if (len(axis_int32_data) == 3):
            assume(
                sorted(axis_int32_data) == [0, 1, 2] and
                axis_int32_data != [0, 1, 2])
            in_shape = draw(st.sampled_from([[C, H, W]]))
        elif (len(axis_int32_data) == 4):
            assume(
                sorted(axis_int32_data) == [0, 1, 2, 3] and
                axis_int32_data != [0, 1, 2, 3])

        def generate_X_data():
            return np.random.normal(0.0, 5.0, in_shape).astype(in_dtype)

        if (target == "Metal" and len(axis_int32_data) == 4):
            for i in range(4):
                for j in range(4):
                    if i != j:
                        assume(in_shape[axis_int32_data.index(i)] *
                               (in_shape[axis_int32_data.index(j)] + 3
                                ) / 4 <= 2048)

        transpose_op = OpConfig(
            type="transpose",
            inputs={"X": ["X_data"]},
            outputs={"Out": ["output_data"]},
            attrs={
                "axis": axis_int32_data,
                "data_format": "AnyLayout",
                "use_mkldnn": use_mkldnn_data,
            })

        program_config = ProgramConfig(
            ops=[transpose_op],
            weights={},
            inputs={
                "X_data": TensorConfig(data_gen=partial(generate_X_data)),
            },
            outputs=["output_data"])
        return program_config

    def sample_predictor_configs(self):
        atol, rtol = 1e-5, 1e-5
        target_str = self.get_target()
        if target_str == "Metal":
            atol, rtol = 5e-4, 5e-4
        return self.get_predictor_configs(), ["transpose"], (atol, rtol)

    def add_ignore_pass_case(self):
        def _teller1(program_config, predictor_config):
            x_shape = list(program_config.inputs["X_data"].shape)
            if predictor_config.target() == TargetType.Metal:
                if x_shape[0] != 1:
                    return True

        self.add_ignore_check_case(
            _teller1, IgnoreReasons.ACCURACY_ERROR,
            "The op output has diff in a specific case on metal. We need to fix it as soon as possible."
        )

        def _teller2(program_config, predictor_config):
            if "nvidia_tensorrt" in self.get_nnadapter_device_name():
                in_shape = program_config.inputs["X_data"].shape
                axis = program_config.ops[0].attrs["axis"]
                if len(in_shape) == 1 or axis[0] != 0:
                    return True

        self.add_ignore_check_case(
            _teller2, IgnoreReasons.PADDLELITE_NOT_SUPPORT,
            "Lite does not support 'in_shape_size == 1' or 'axis[0] != 0' on nvidia_tensorrt."
        )

    def test(self, *args, **kwargs):
        self.run_and_statis(quant=False, max_examples=25)


if __name__ == "__main__":
    unittest.main(argv=[''])
