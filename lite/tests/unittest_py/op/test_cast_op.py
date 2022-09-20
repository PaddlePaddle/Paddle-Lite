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

from os import truncate
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


class TestCastOp(AutoScanTest):
    def __init__(self, *args, **kwargs):
        AutoScanTest.__init__(self, *args, **kwargs)
        self.enable_testing_on_place(
            TargetType.Host,
            PrecisionType.FP32,
            DataLayoutType.NCHW,
            thread=[1, 4])
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
        self.enable_devices_on_nnadapter(device_names=[
            "cambricon_mlu", "nvidia_tensorrt", "intel_openvino",
            "kunlunxin_xtcl"
        ])

    def is_program_valid(self,
                         program_config: ProgramConfig,
                         predictor_config: CxxConfig) -> bool:
        return True

    def sample_program_configs(self, draw):
        in_shape = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=10), min_size=1, max_size=4))
        # BOOL = 0;INT16 = 1;INT32 = 2;INT64 = 3;FP16 = 4;FP32 = 5;FP64 = 6;
        in_dtype = draw(st.sampled_from([0, 2, 3, 5]))
        out_dtype = draw(st.sampled_from([0, 2, 3, 5]))

        # BOOL and INT16 and FP16 and FP64 paddle-lite doesn't support
        def generate_input(*args, **kwargs):
            if in_dtype == 0:
                return np.random.random(in_shape).astype(np.bool)
            elif in_dtype == 1:
                return np.random.random(in_shape).astype(np.int16)
            elif in_dtype == 2:
                return np.random.random(in_shape).astype(np.int32)
            elif in_dtype == 3:
                return np.random.random(in_shape).astype(np.int64)
            elif in_dtype == 4:
                return np.random.random(in_shape).astype(np.float16)
            elif in_dtype == 5:
                return np.random.random(in_shape).astype(np.float32)
            else:
                print("in_dtype is error! ", in_dtype)

        cast_op = OpConfig(
            type="cast",
            inputs={"X": ["input_data"]},
            outputs={"Out": ["output_data"]},
            attrs={"in_dtype": in_dtype,
                   "out_dtype": out_dtype})

        program_config = ProgramConfig(
            ops=[cast_op],
            weights={},
            inputs={
                "input_data": TensorConfig(data_gen=partial(generate_input))
            },
            outputs=["output_data"])

        in_dtype = program_config.ops[0].attrs["in_dtype"]
        out_dtype = program_config.ops[0].attrs["out_dtype"]
        assume(in_dtype != 0)
        assume(out_dtype != 0)
        return program_config

    def sample_predictor_configs(self):
        atol, rtol = 1e-5, 1e-5
        target_str = self.get_target()
        if target_str == "Metal":
            atol, rtol = 5e-4, 5e-4
        return self.get_predictor_configs(), ["cast"], (atol, rtol)

    def add_ignore_pass_case(self):
        def _teller1(program_config, predictor_config):
            x_shape = list(program_config.inputs["input_data"].shape)
            in_dtype = program_config.ops[0].attrs["in_dtype"]
            out_dtype = program_config.ops[0].attrs["out_dtype"]
            if predictor_config.target() == TargetType.Metal:
                if len(x_shape) != 4 or in_dtype != 5 or out_dtype != 5:
                    return True

        self.add_ignore_check_case(
            _teller1, IgnoreReasons.PADDLELITE_NOT_SUPPORT,
            "The op output has diff in a specific case on metal. We need to fix it as soon as possible."
        )

        def _teller2(program_config, predictor_config):
            if self.get_nnadapter_device_name() == "nvidia_tensorrt":
                in_shape = program_config.inputs["input_data"].shape
                in_dtype = program_config.ops[0].attrs["in_dtype"]
                out_dtype = program_config.ops[0].attrs["out_dtype"]
                if len(in_shape) == 1 \
                    or [in_dtype, out_dtype] not in [[2, 5], [5, 2]]:
                    return True

        self.add_ignore_check_case(
            _teller2, IgnoreReasons.PADDLELITE_NOT_SUPPORT,
            "nvidia_tensorrt now support int32<->float32.")

        def _teller3(program_config, predictor_config):
            if self.get_nnadapter_device_name() == "kunlunxin_xtcl":
                in_dtype = program_config.ops[0].attrs["in_dtype"]
                out_dtype = program_config.ops[0].attrs["out_dtype"]
                if in_dtype == out_dtype:
                    return True

        self.add_ignore_check_case(
            _teller3, IgnoreReasons.PADDLELITE_NOT_SUPPORT,
            "Lite not support same dtype cast on kunlunxin_xtcl")

    def test(self, *args, **kwargs):
        target_str = self.get_target()
        max_examples = 250
        if target_str == "Metal":
            # Make sure to generate enough valid cases for Metal
            max_examples = 1000
        self.run_and_statis(quant=False, max_examples=max_examples)


if __name__ == "__main__":
    unittest.main(argv=[''])
