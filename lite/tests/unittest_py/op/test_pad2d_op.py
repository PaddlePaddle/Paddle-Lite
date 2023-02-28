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
import argparse

import numpy as np
from functools import partial


class TestPad2dOp(AutoScanTest):
    def __init__(self, *args, **kwargs):
        AutoScanTest.__init__(self, *args, **kwargs)
        self.enable_testing_on_place(
            TargetType.Host,
            PrecisionType.FP32,
            DataLayoutType.NCHW,
            thread=[1, 2])
        # When Host and X86 exist at the same time, the error occurred.
        # self.enable_testing_on_place(
        #     TargetType.X86,
        #     PrecisionType.FP32,
        #     DataLayoutType.NCHW,
        #     thread=[1, 2])
        self.enable_testing_on_place(
            TargetType.ARM,
            PrecisionType.FP32,
            DataLayoutType.NCHW,
            thread=[1, 4])
        opencl_places = [
            Place(TargetType.OpenCL, PrecisionType.FP16,
                  DataLayoutType.ImageDefault),
            Place(TargetType.OpenCL, PrecisionType.FP32, DataLayoutType.NCHW),
            Place(TargetType.OpenCL, PrecisionType.FP16,
                  DataLayoutType.ImageFolder), Place(
                      TargetType.OpenCL, PrecisionType.Any,
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
                  DataLayoutType.MetalTexture2DArray)
        ]
        self.enable_testing_on_place(places=metal_places)
        self.enable_testing_on_place(
            TargetType.ARM,
            PrecisionType.FP16,
            DataLayoutType.NCHW,
            thread=[1, 4])
        self.enable_testing_on_place(TargetType.NNAdapter, PrecisionType.FP32)
        self.enable_devices_on_nnadapter(
            device_names=["intel_openvino", "kunlunxin_xtcl"])

    def is_program_valid(self,
                         program_config: ProgramConfig,
                         predictor_config: CxxConfig) -> bool:
        return True

    def sample_program_configs(self, draw):
        in_shape = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=32), min_size=4, max_size=4))
        mode = draw(st.sampled_from(["constant", "reflect", "edge"]))
        value_data = draw(st.floats(min_value=0.0, max_value=4.0))
        padding_data = draw(
            st.lists(
                st.integers(
                    min_value=0, max_value=4), min_size=4, max_size=4))
        data_format = draw(st.sampled_from(["NCHW", "NHWC"]))
        for i in range(4):
            assume(padding_data[i] < in_shape[1])
            assume(padding_data[i] < in_shape[2])
            assume(padding_data[i] < in_shape[3])

        def generate_input(*args, **kwargs):
            return np.random.random(in_shape).astype(np.float32)

        def generate_paddings(*args, **kwargs):
            return np.array(padding_data).astype(np.int32)

        build_ops = OpConfig(
            type="pad2d",
            inputs={
                "X": ["input_data"],
                #"Paddings": ["paddings_data"]
            },
            outputs={"Out": ["output_data"], },
            attrs={
                "paddings": padding_data,
                "mode": mode,
                "pad_value": value_data,
                "data_format": data_format,
            })
        program_config = ProgramConfig(
            ops=[build_ops],
            weights={
                #"paddings_data": TensorConfig(data_gen=partial(generate_paddings))
            },
            inputs={
                "input_data": TensorConfig(data_gen=partial(generate_input))
            },
            outputs=["output_data"])
        return program_config

    def sample_predictor_configs(self):
        atol, rtol = 1e-5, 1e-5
        target_str = self.get_target()
        if target_str == "Metal":
            atol, rtol = 1e-3, 1e-3
        return self.get_predictor_configs(), ["pad2d"], (atol, rtol)

    def add_ignore_pass_case(self):
        def _teller1(program_config, predictor_config):
            target_type = predictor_config.target()
            data_format = program_config.ops[0].attrs["data_format"]
            if target_type in [TargetType.OpenCL, TargetType.Metal
                               ] and data_format == "NHWC":
                return True

        self.add_ignore_check_case(
            _teller1, IgnoreReasons.ACCURACY_ERROR,
            "Lite doesn't not support for NHWC pad2d on ARM && fp16, and on Metal and OpenCL"
        )

        def _teller2(program_config, predictor_config):
            if self.get_nnadapter_device_name() == "kunlunxin_xtcl":
                mode = program_config.ops[0].attrs["mode"]
                if mode != "constant":
                    return True

        self.add_ignore_check_case(
            _teller2, IgnoreReasons.PADDLELITE_NOT_SUPPORT,
            "Lite only support constant mode on kunlunxin_xtcl")

    def test(self, *args, **kwargs):
        self.run_and_statis(quant=False, max_examples=100)


if __name__ == "__main__":
    unittest.main(argv=[''])
