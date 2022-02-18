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


class TestUnsqueeze2Op(AutoScanTest):
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
        self.enable_devices_on_nnadapter(device_names=["cambricon_mlu"])

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

        def generate_XShape_data():
            return np.random.random([6]).astype(np.float32)

        unsqueeze2_op = OpConfig(
            type="unsqueeze2",
            inputs=inputs,
            outputs={"Out": ["Out_data"],
                     "XShape": ["XShape_data"]},
            attrs={"axes": axes_data, })
        unsqueeze2_op.outputs_dtype = {"Out_data": in_dtype}

        program_config = ProgramConfig(
            ops=[unsqueeze2_op],
            weights={
                "XShape_data":
                TensorConfig(data_gen=partial(generate_XShape_data))
            },
            inputs={
                "X_data": TensorConfig(data_gen=partial(generate_X_data)),
                "AxesTensor_data":
                TensorConfig(data_gen=partial(generate_AxesTensor_data)),
                "AxesTensorList_data":
                TensorConfig(data_gen=partial(generate_AxesTensorList_data))
            },
            outputs=["Out_data"])
        # OpenCL Check failed so remove "XShape_data"
        return program_config

    def sample_predictor_configs(self):
        return self.get_predictor_configs(), [""], (1e-5, 1e-5)

    def add_ignore_pass_case(self):
        pass

    def test(self, *args, **kwargs):
        self.run_and_statis(quant=False, max_examples=25)


if __name__ == "__main__":
    unittest.main(argv=[''])
