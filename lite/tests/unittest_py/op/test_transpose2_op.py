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


class TestTranspose2Op(AutoScanTest):
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
        self.enable_testing_on_place(TargetType.NNAdapter, PrecisionType.FP32)
        self.enable_devices_on_nnadapter(device_names=[
            "kunlunxin_xtcl", "nvidia_tensorrt", "intel_openvino"
        ])

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

        # segmentation fault
        metal_places = [
            Place(TargetType.Metal, PrecisionType.FP32,
                  DataLayoutType.MetalTexture2DArray),
            Place(TargetType.Metal, PrecisionType.FP16,
                  DataLayoutType.MetalTexture2DArray),
            Place(TargetType.ARM, PrecisionType.FP32),
            Place(TargetType.Host, PrecisionType.FP32)
        ]
        self.enable_testing_on_place(places=metal_places)

    def is_program_valid(self,
                         program_config: ProgramConfig,
                         predictor_config: CxxConfig) -> bool:
        return True

    def sample_program_configs(self, draw):
        N = draw(st.integers(min_value=1, max_value=4))
        C = draw(st.integers(min_value=1, max_value=17))
        H = draw(st.integers(min_value=1, max_value=17))
        W = draw(st.integers(min_value=1, max_value=17))
        in_shape = draw(st.sampled_from([[N, C, H, W], []]))
        target = self.get_target()
        use_mkldnn_data = False
        in_dtype = np.float32
        if (target == "X86"):
            use_mkldnn_data = True
            in_dtype = draw(st.sampled_from([np.float32]))
        elif (target == "ARM"):
            in_dtype = draw(
                st.sampled_from([np.float32, np.int32,
                                 np.int64]))  # paddle doesn't have int8
            # ToDo : 
            # fp16 can not be verified
        elif (target in ["OpenCL", "Metal"]):
            in_dtype = draw(st.sampled_from([np.float32]))
            # ToDo : 
            # fp16 can not be verified

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

        if in_shape == []:
            axis_int32_data = []

        def generate_X_data():
            return np.random.random(in_shape).astype(in_dtype)

        outputs = {"Out": ["output_data"]}
        outputs_data = ["output_data"]
        if self.get_target() == "NNAdapter":
            outputs = {"Out": ["output_data"]}
            outputs_data = ["output_data"]
        transpose2_op = OpConfig(
            type="transpose2",
            inputs={"X": ["X_data"]},
            outputs=outputs,
            attrs={
                "axis": axis_int32_data,
                "data_format": "AnyLayout",
                "use_mkldnn": use_mkldnn_data,
            })
        transpose2_op.outputs_dtype = {"output_data": in_dtype}
        program_config = ProgramConfig(
            ops=[transpose2_op],
            weights={},
            inputs={
                "X_data": TensorConfig(data_gen=partial(generate_X_data)),
            },
            outputs=outputs_data)
        return program_config

    def sample_predictor_configs(self):
        return self.get_predictor_configs(), [""], (1e-5, 1e-5)

    def add_ignore_pass_case(self):
        def teller1(program_config, predictor_config):
            if "nvidia_tensorrt" in self.get_nnadapter_device_name():
                in_shape = program_config.inputs["X_data"].shape
                axis = program_config.ops[0].attrs["axis"]
                if len(in_shape) == 1 or axis[0] != 0:
                    return True

        self.add_ignore_check_case(
            teller1, IgnoreReasons.PADDLELITE_NOT_SUPPORT,
            "Lite does not support 'in_shape_size == 1' or 'axis[0] != 0' on nvidia_tensorrt."
        )

        def _teller2(program_config, predictor_config):
            target_type = predictor_config.target()
            in_x_shape = list(program_config.inputs["X_data"].shape)
            if target_type not in [
                    TargetType.ARM, TargetType.Host, TargetType.Metal,
                    TargetType.X86
            ]:
                if len(in_x_shape) == 0:
                    return True

        self.add_ignore_check_case(
            _teller2, IgnoreReasons.PADDLELITE_NOT_SUPPORT,
            "Only test 0D-tensor on CPU(ARM/Host/X86/Metal/OpenCL) now.")

        def _teller3(program_config, predictor_config):
            target_type = predictor_config.target()
            if target_type == TargetType.Metal:
                return True

        self.add_ignore_check_case(_teller3,
                                   IgnoreReasons.PADDLELITE_NOT_SUPPORT,
                                   "Lite has diff on Metal")

    def test(self, *args, **kwargs):
        self.run_and_statis(quant=False, max_examples=25)


if __name__ == "__main__":
    unittest.main(argv=[''])
