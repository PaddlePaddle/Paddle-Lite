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
import numpy as np
from functools import partial


class TestFlattenOp(AutoScanTest):
    def __init__(self, *args, **kwargs):
        AutoScanTest.__init__(self, *args, **kwargs)
        self.enable_testing_on_place(TargetType.Host, PrecisionType.FP32,
                                     DataLayoutType.NCHW)
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
        self.enable_devices_on_nnadapter(
            device_names=["cambricon_mlu", "intel_openvino"])

    def is_program_valid(self,
                         program_config: ProgramConfig,
                         predictor_config: CxxConfig) -> bool:
        return True

    def sample_program_configs(self, draw):
        in_shape = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=8), min_size=2, max_size=4))

        def generate_input(*args, **kwargs):
            if kwargs["type"] == "int32":
                return np.random.randint(kwargs["low"], kwargs["high"],
                                         kwargs["shape"]).astype(np.int32)
            elif kwargs["type"] == "int64":
                return np.random.randint(kwargs["low"], kwargs["high"],
                                         kwargs["shape"]).astype(np.int64)
            elif kwargs["type"] == "float32":
                return (kwargs["high"] - kwargs["low"]) * np.random.random(
                    kwargs["shape"]).astype(np.float32) + kwargs["low"]

        input_type = draw(st.sampled_from(["float32", "int64", "int32"]))

        #wait for atuo_scan_base bug fix
        input_type = "float32"

        axis = draw(st.integers(min_value=0, max_value=len(in_shape) - 1))

        flatten_op = OpConfig(
            type="flatten",
            inputs={"X": ["input_data"]},
            outputs={"Out": ["output_data"]},
            attrs={"axis": axis})

        program_config = ProgramConfig(
            ops=[flatten_op],
            weights={},
            inputs={
                "input_data": TensorConfig(data_gen=partial(
                    generate_input,
                    type=input_type,
                    low=-10,
                    high=10,
                    shape=in_shape))
            },
            outputs=["output_data"])

        return program_config

    def sample_predictor_configs(self):
        atol, rtol = 1e-5, 1e-5
        target_str = self.get_target()
        if target_str == "Metal":
            atol, rtol = 1e-3, 1e-3

        return self.get_predictor_configs(), ["flatten"], (atol, rtol)

    def add_ignore_pass_case(self):
        def _teller1(program_config, predictor_config):
            target_type = predictor_config.target()
            in_shape = list(program_config.inputs["input_data"].shape)
            axis = program_config.ops[0].attrs["axis"]
            if target_type == TargetType.Metal:
                if len(in_shape) != 4 \
                    or in_shape[0] != 1 \
                    or axis != 1:
                    return True

        self.add_ignore_check_case(
            _teller1, IgnoreReasons.PADDLELITE_NOT_SUPPORT,
            "Lite does not support this op in a specific case on metal. We need to fix it as soon as possible."
        )

        def _teller2(program_config, predictor_config):
            target_type = predictor_config.target()
            in_shape = list(program_config.inputs["input_data"].shape)
            axis = program_config.ops[0].attrs["axis"]
            if target_type == TargetType.NNAdapter:
                if axis == 1 and len(in_shape) == 2:
                    return True

        self.add_ignore_check_case(
            _teller2, IgnoreReasons.PADDLELITE_NOT_SUPPORT,
            "Lite does not support this op when 'axis = 1 and len(in_shape) = 2' on nnadapter."
        )

    def test(self, *args, **kwargs):
        target_str = self.get_target()
        max_examples = 100
        if target_str == "Metal":
            # Make sure to generate enough valid cases for Metal
            max_examples = 3000
        self.run_and_statis(quant=False, max_examples=max_examples)


if __name__ == "__main__":
    unittest.main(argv=[''])
