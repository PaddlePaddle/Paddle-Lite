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


class TestUnsqueezeOp(AutoScanTest):
    def __init__(self, *args, **kwargs):
        AutoScanTest.__init__(self, *args, **kwargs)
        host_places = [
            Place(TargetType.Host, PrecisionType.FP32, DataLayoutType.NCHW)
        ]
        self.enable_testing_on_place(places=host_places)

        # opencl demo
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
        self.enable_devices_on_nnadapter(device_names=[
            "cambricon_mlu", "nvidia_tensorrt", "intel_openvino",
            "kunlunxin_xtcl"
        ])

        # metal errors
        # so I comment them
        # metal_places = [
        #     Place(TargetType.Metal, PrecisionType.FP32,
        #           DataLayoutType.MetalTexture2DArray),
        #     Place(TargetType.Metal, PrecisionType.FP16,
        #           DataLayoutType.MetalTexture2DArray),
        #     Place(TargetType.ARM, PrecisionType.FP32),
        #     Place(TargetType.Host, PrecisionType.FP32)
        # ]
        # self.enable_testing_on_place(places=metal_places)

    def is_program_valid(self,
                         program_config: ProgramConfig,
                         predictor_config: CxxConfig) -> bool:
        if predictor_config.target() == TargetType.NNAdapter:
            if "AxesTensor" in program_config.ops[0].inputs:
                return False
        return True

    def sample_program_configs(self, draw):
        N = draw(st.integers(min_value=1, max_value=4))
        C = draw(st.integers(min_value=1, max_value=128))
        H = draw(st.integers(min_value=1, max_value=128))
        W = draw(st.integers(min_value=1, max_value=128))
        in_shape = draw(st.sampled_from([[N, C, H, W]]))
        in_dtype = np.float32
        target = self.get_target()

        if (target in ["X86", "ARM"]):
            in_dtype = draw(st.sampled_from([np.float32, np.int32, np.int64]))
        elif (target in ["OpenCL", "Metal"]):
            in_dtype = draw(st.sampled_from([np.float32]))

        def generate_X_data():
            return np.random.normal(0.0, 5.0, in_shape).astype(in_dtype)

        axes_data = draw(
            st.lists(
                st.integers(
                    min_value=0, max_value=3), min_size=1, max_size=2))
        inputs = {"X": ["X_data"]}
        choose_axes = draw(
            st.sampled_from(["axes", "AxesTensor", "AxesTensorList"]))

        def generate_AxesTensor_data():
            if (choose_axes == "AxesTensor"):
                inputs["AxesTensor"] = ["AxesTensor_data"]
                return np.array(axes_data).astype(np.int32)
            else:
                return np.random.randint(1, 5, []).astype(np.int32)

        def generate_AxesTensorList_data():
            if (choose_axes == "AxesTensorList"):
                #inputs["AxesTensorList"] = ["AxesTensorList_data"]
                return np.array(axes_data).astype(np.int32)
            else:
                return np.random.randint(1, 5, []).astype(np.int32)

        unsqueeze_op = OpConfig(
            type="unsqueeze",
            inputs=inputs,
            outputs={"Out": ["Out_data"]},
            attrs={"axes": axes_data, })
        unsqueeze_op.outputs_dtype = {"Out_data": in_dtype}

        program_config = ProgramConfig(
            ops=[unsqueeze_op],
            weights={},
            inputs={
                "X_data": TensorConfig(data_gen=partial(generate_X_data)),
                "AxesTensor_data":
                TensorConfig(data_gen=partial(generate_AxesTensor_data)),
                "AxesTensorList_data":
                TensorConfig(data_gen=partial(generate_AxesTensorList_data))
            },
            outputs=["Out_data"])
        return program_config

    def sample_predictor_configs(self):
        return self.get_predictor_configs(), [""], (1e-5, 1e-5)

    def add_ignore_pass_case(self):
        def teller1(program_config, predictor_config):
            if "nvidia_tensorrt" in self.get_nnadapter_device_name():
                in_shape = program_config.inputs["X_data"].shape
                axes = program_config.ops[0].attrs["axes"]
                if len(in_shape) == 1 or 0 in axes:
                    return True

        self.add_ignore_check_case(
            teller1, IgnoreReasons.PADDLELITE_NOT_SUPPORT,
            "Lite does not support 'in_shape_size == 1' or '0 in axes' on nvidia_tensorrt."
        )

    def test(self, *args, **kwargs):
        target_str = self.get_target()
        max_examples = 25
        if target_str == "NNAdapter":
            # Make sure to generate enough valid cases for NNAdapter
            max_examples = 200
        self.run_and_statis(quant=False, max_examples=max_examples)


if __name__ == "__main__":
    unittest.main(argv=[''])
