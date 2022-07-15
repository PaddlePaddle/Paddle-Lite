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


class TestNearestV2InterpOp(AutoScanTest):
    def __init__(self, *args, **kwargs):
        AutoScanTest.__init__(self, *args, **kwargs)
        self.enable_testing_on_place(TargetType.NNAdapter, PrecisionType.FP32)
        self.enable_devices_on_nnadapter(device_names=[
            "cambricon_mlu", "nvidia_tensorrt", "intel_openvino"
        ])
        # precision bugs will be fix in the future
        self.enable_testing_on_place(
            TargetType.ARM, [PrecisionType.FP16, PrecisionType.FP32],
            DataLayoutType.NCHW,
            thread=[1, 4])
        self.enable_testing_on_place(
            TargetType.X86,
            PrecisionType.FP32,
            DataLayoutType.NCHW,
            thread=[1, 4])
        self.enable_testing_on_place(
            TargetType.Metal,
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
        return True

    def sample_program_configs(self, draw):
        in_num = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=4), min_size=1, max_size=1))
        in_c_h_w = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=64), min_size=3, max_size=3))
        X_shape = in_num + in_c_h_w
        align_corners = draw(st.booleans())
        scale1 = draw(st.floats(min_value=0.1, max_value=10.0))
        scale2 = draw(st.floats(min_value=0.1, max_value=10.0))
        interp_method = draw(st.sampled_from(["nearest"]))
        out_w = draw(st.integers(min_value=1, max_value=32))
        out_h = draw(st.integers(min_value=1, max_value=32))
        data_layout = draw(st.sampled_from(["NCHW"]))
        test_case = draw(st.sampled_from([1, 2, 3]))
        infer_shape_with_scale = draw(st.booleans())

        def generate_input1(*args, **kwargs):
            return np.random.normal(0.0, 1.0, X_shape).astype(np.float32)

        def generate_scale(*args, **kwargs):
            tmp = np.random.normal(0.1, 10.0, 2).astype(np.float32)
            assume(tmp[0] * X_shape[2] > 1.0)
            assume(tmp[0] * X_shape[3] > 1.0)
            assume(tmp[1] * X_shape[2] > 1.0)
            assume(tmp[1] * X_shape[3] > 1.0)
            return tmp

        def generate_input2(*args, **kwargs):
            return np.random.randint(1, 32, 2).astype(np.int32)

        def generate_input1_fp16(*args, **kwargs):
            return np.random.normal(0.0, 1.0, X_shape).astype(np.float16)

        assume(scale1 * X_shape[2] > 1.0)
        assume(scale1 * X_shape[3] > 1.0)
        assume(scale2 * X_shape[2] > 1.0)
        assume(scale2 * X_shape[3] > 1.0)
        assume(test_case != 2)
        scale = [scale1, scale2]

        if "intel_openvino" in self.get_nnadapter_device_name():
            assume(scale1 > 1.5 or scale1 < 1)
            assume(scale2 > 1.5 or scale2 < 1)
            if infer_shape_with_scale:
                out_h = -1
                out_w = -1
            else:
                scale = []

        nnadapter_device_name = self.get_nnadapter_device_name()
        has_tensorrt_device = "nvidia_tensorrt" in self.get_nnadapter_device_name(
        )
        if self.get_target() == 'NNAdapter':
            nearest_interp_v2 = OpConfig(
                type="nearest_interp_v2",
                inputs={"X": ["input_data_x"]},
                outputs={"Out": ["output_data"]},
                attrs={
                    "data_layout": data_layout,
                    "scale": scale,
                    "out_w": out_w,
                    "out_h": out_h,
                    "interp_method": interp_method,
                    "align_corners": False
                    if has_tensorrt_device else align_corners
                })
            program_config = ProgramConfig(
                ops=[nearest_interp_v2],
                weights={},
                inputs={
                    "input_data_x": TensorConfig(data_gen=generate_input1)
                },
                outputs={"output_data"})
            return program_config

        if test_case == 1:
            nearest_interp_v2 = OpConfig(
                type="nearest_interp_v2",
                inputs={"X": ["input_data_x"],
                        "Scale": ["Scale"]},
                outputs={"Out": ["output_data"]},
                attrs={
                    "data_layout": data_layout,
                    "scale": [scale1, scale2],
                    "out_w": out_w,
                    "out_h": out_h,
                    "interp_method": interp_method,
                    "align_corners": align_corners
                })
            program_config = ProgramConfig(
                ops=[nearest_interp_v2],
                weights={},
                inputs={
                    "input_data_x": TensorConfig(data_gen=generate_input1),
                    "Scale": TensorConfig(data_gen=generate_scale)
                },
                outputs={"output_data"})
        elif test_case == 2:
            nearest_interp_v2 = OpConfig(
                type="nearest_interp_v2",
                inputs={
                    "X": ["input_data_x"],
                    "Scale": ["Scale"],
                    "OutSize": ["OutSize"]
                },
                outputs={"Out": ["output_data"]},
                attrs={
                    "data_layout": data_layout,
                    "scale": [scale1, scale2],
                    "out_w": out_w,
                    "out_h": out_h,
                    "interp_method": interp_method,
                    "align_corners": align_corners
                })
            program_config = ProgramConfig(
                ops=[nearest_interp_v2],
                weights={},
                inputs={
                    "input_data_x": TensorConfig(data_gen=generate_input1),
                    "Scale": TensorConfig(data_gen=generate_scale),
                    "OutSize": TensorConfig(data_gen=generate_input2)
                },
                outputs={"output_data"})
        else:
            nearest_interp_v2 = OpConfig(
                type="nearest_interp_v2",
                inputs={"X": ["input_data_x"]},
                outputs={"Out": ["output_data"]},
                attrs={
                    "data_layout": data_layout,
                    "scale": [scale1, scale2],
                    "out_w": out_w,
                    "out_h": out_h,
                    "interp_method": interp_method,
                    "align_corners": align_corners
                })
            program_config = ProgramConfig(
                ops=[nearest_interp_v2],
                weights={},
                inputs={
                    "input_data_x": TensorConfig(data_gen=generate_input1)
                },
                outputs={"output_data"})
        return program_config

    def sample_predictor_configs(self):
        return self.get_predictor_configs(), ["nearest_interp_v2"], (1e-5,
                                                                     1e-5)

    def add_ignore_pass_case(self):
        def _teller1(program_config, predictor_config):
            if predictor_config.target() == TargetType.Metal:
                return True

        def _teller3(program_config, predictor_config):
            nnadapter_device_name = self.get_nnadapter_device_name()
            if nnadapter_device_name == "nvidia_tensorrt":
                return True

        self.add_ignore_check_case(
            _teller1, IgnoreReasons.PADDLELITE_NOT_SUPPORT,
            "Lite does not support this op in a specific case on metal. We need to fix it as soon as possible."
        )
        self.add_ignore_check_case(
            _teller3, IgnoreReasons.PADDLELITE_NOT_SUPPORT,
            "The paddle's and trt_layer's results has diff in a specific case. We need to fix it as soon as possible."
        )

    def test(self, *args, **kwargs):
        self.run_and_statis(quant=False, max_examples=200)


if __name__ == "__main__":
    unittest.main(argv=[''])
