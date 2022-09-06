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


class TestPool2dOp(AutoScanTest):
    def __init__(self, *args, **kwargs):
        AutoScanTest.__init__(self, *args, **kwargs)
        self.enable_testing_on_place(
            TargetType.X86,
            PrecisionType.FP32,
            DataLayoutType.NCHW,
            thread=[1, 4])
        self.enable_testing_on_place(
            TargetType.ARM,
            PrecisionType.FP32,
            DataLayoutType.NCHW,
            thread=[1, 4])
        self.enable_testing_on_place(
            TargetType.ARM,
            PrecisionType.FP16,
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
        self.enable_devices_on_nnadapter(
            device_names=["cambricon_mlu", "nvidia_tensorrt"])

    def is_program_valid(self,
                         program_config: ProgramConfig,
                         predictor_config: CxxConfig) -> bool:
        return True

    def sample_program_configs(self, draw):
        in_shape = draw(
            st.lists(
                st.integers(
                    min_value=4, max_value=128),
                min_size=3,
                max_size=3))
        in_shape.insert(0, draw(st.integers(min_value=1, max_value=4)))
        ksize = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=7), min_size=2, max_size=2))
        strides = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=16), min_size=2, max_size=2))
        paddings = draw(
            st.lists(
                st.integers(
                    min_value=0, max_value=16), min_size=2, max_size=2))
        padding_algorithm = draw(
            st.sampled_from(["EXPLICIT", "VALID", "SAME"]))
        pooling_type = draw(st.sampled_from(["max", "avg"]))
        global_pooling = draw(st.booleans())
        exclusive = draw(st.booleans())
        ceil_mode = draw(st.booleans())
        adaptive = draw(st.booleans())
        use_cudnn = False
        use_mkldnn = False
        use_quantizer = False
        is_test = True
        data_format = "NCHW"

        assume(ksize[0] <= in_shape[2])
        assume(ksize[1] <= in_shape[3])
        if adaptive == False:
            assume(ksize[0] > paddings[0] and ksize[1] > paddings[1])
        if adaptive == False and ceil_mode == True:
            assume(strides[0] > 1 and strides[1] > 1)

        def generate_input(*args, **kwargs):
            return np.random.normal(0.0, 1.0, in_shape).astype(np.float32)

        build_ops = OpConfig(
            type="pool2d",
            inputs={"X": ["input_data"]},
            outputs={"Out": ["output_data"]},
            attrs={
                "pooling_type": pooling_type,
                "ksize": ksize,
                "global_pooling": global_pooling,
                "strides": strides,
                "paddings": paddings,
                "exclusive": exclusive,
                "adaptive": adaptive,
                "use_cudnn": use_cudnn,
                "ceil_mode": ceil_mode,
                "use_mkldnn": use_mkldnn,
                "use_quantizer": use_quantizer,
                "is_test": is_test,
                "padding_algorithm": padding_algorithm,
                "data_format": data_format
            })
        program_config = ProgramConfig(
            ops=[build_ops],
            weights={},
            inputs={
                "input_data": TensorConfig(data_gen=partial(generate_input))
            },
            outputs=["output_data"])
        return program_config

    def sample_predictor_configs(self):
        atol, rtol = 1e-5, 1e-5
        config_lists = self.get_predictor_configs()
        for config in config_lists:
            if config.precision() in [PrecisionType.FP16]:
                atol, rtol = 1e-3, 1e-3
            if config.target() in [TargetType.Metal]:
                atol, rtol = 5e-4, 5e-4
        return self.get_predictor_configs(), ["pool2d"], (atol, rtol)

    def add_ignore_pass_case(self):
        def teller1(program_config, predictor_config):
            if predictor_config.target() == TargetType.Metal:
                if program_config.ops[0].attrs["padding_algorithm"] == "SAME" \
                    or program_config.ops[0].attrs["pooling_type"] == "avg" :
                    return True
                strides = program_config.ops[0].attrs["strides"]
                if program_config.ops[0].attrs["ceil_mode"] == True \
                    and strides[0] != strides[1]:
                    return True

        self.add_ignore_check_case(
            teller1, IgnoreReasons.ACCURACY_ERROR,
            "The op output has diff in a specific case. We need to fix it as soon as possible."
        )

        def teller2(program_config, predictor_config):
            in_shape = list(program_config.inputs["input_data"].shape)
            if predictor_config.target() == TargetType.Metal:
                return True
                if program_config.ops[0].attrs["adaptive"] == True \
                    or program_config.ops[0].attrs["ceil_mode"] == True:
                    return True
                if in_shape[0] != 1:
                    return True

        self.add_ignore_check_case(
            teller2, IgnoreReasons.PADDLELITE_NOT_SUPPORT,
            "Lite does not support this op in a specific case on metal. We need to fix it as soon as possible."
        )

        def teller3(program_config, predictor_config):
            if predictor_config.target() == TargetType.ARM:
                return True

        self.add_ignore_check_case(
            teller3, IgnoreReasons.PADDLE_NOT_SUPPORT,
            "Paddle does not support this op in a specific case. We have fedback to the Paddle developer."
        )

        def teller4(program_config, predictor_config):
            if "nvidia_tensorrt" in self.get_nnadapter_device_name():
                if program_config.ops[0].attrs["adaptive"] == True \
                    or program_config.ops[0].attrs["ceil_mode"] == True:
                    return True

        self.add_ignore_check_case(
            teller4, IgnoreReasons.PADDLELITE_NOT_SUPPORT,
            "Lite does not support 'adaptive == True' or 'ceil_mode == True' on nvidia_tensorrt."
        )

    def test(self, *args, **kwargs):
        target_str = self.get_target()
        max_examples = 100
        if target_str == "OpenCL":
            # Make sure to generate enough valid cases for OpenCL
            max_examples = 200
        if target_str == "Metal":
            # Make sure to generate enough valid cases for Metal
            max_examples = 500
        self.run_and_statis(quant=False, max_examples=max_examples)


if __name__ == "__main__":
    unittest.main(argv=[''])
