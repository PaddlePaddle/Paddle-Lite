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
from functools import reduce


class TestReshape2Op(AutoScanTest):
    def __init__(self, *args, **kwargs):
        AutoScanTest.__init__(self, *args, **kwargs)
        self.enable_testing_on_place(
            TargetType.Host,
            PrecisionType.FP32,
            DataLayoutType.NCHW,
            thread=[1, 2])

        # opencl
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

        # metal
        metal_places = [
            Place(TargetType.Metal, PrecisionType.FP32,
                  DataLayoutType.MetalTexture2DArray),
            Place(TargetType.Metal, PrecisionType.FP16,
                  DataLayoutType.MetalTexture2DArray)
        ]
        self.enable_testing_on_place(places=metal_places)
        self.enable_testing_on_place(TargetType.NNAdapter, PrecisionType.FP32)
        self.enable_devices_on_nnadapter(device_names=[
            "kunlunxin_xtcl", "cambricon_mlu", "nvidia_tensorrt",
            "intel_openvino"
        ])

    def is_program_valid(self,
                         program_config: ProgramConfig,
                         predictor_config: CxxConfig) -> bool:
        return True

    def sample_program_configs(self, draw):
        in_shape = draw(
            st.lists(
                st.integers(
                    min_value=5, max_value=10), min_size=4, max_size=4))
        attr_shape = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=max(in_shape)),
                min_size=1,
                max_size=len(in_shape)))

        shape_tensor = []
        for i in range(len(attr_shape) - 1, -1, -1):
            shape_tensor.append(attr_shape[i])
        assume(
            reduce(lambda x, y: x * y, attr_shape) == reduce(
                lambda x, y: x * y, in_shape))

        in_shape = draw(st.sampled_from([in_shape, []]))
        if in_shape == []:
            attr_shape = [1]
            shape_tensor = [1, 1]

        # The parameter shape in ReshapeOp must be set
        with_shape_attr = draw(st.sampled_from([True]))
        with_shape_tensor = draw(st.sampled_from([True, False]))

        def generate_input(*args, **kwargs):
            return np.random.random(in_shape).astype(np.float32)

        def generate_shape(*args, **kwargs):
            return np.asarray(shape_tensor).astype(np.int32)

        if (with_shape_attr and with_shape_tensor):
            build_ops = OpConfig(
                type="reshape2",
                inputs={"X": ["input_data"],
                        "Shape": ["input_shape"]},
                outputs={
                    "Out": ["output_data"],
                    "XShape": ["x_shape"],
                },
                attrs={"shape": attr_shape, })
            program_config = ProgramConfig(
                ops=[build_ops],
                weights={},
                inputs={
                    "input_data":
                    TensorConfig(data_gen=partial(generate_input)),
                    "input_shape":
                    TensorConfig(data_gen=partial(generate_shape)),
                },
                outputs=["output_data"])
        elif (with_shape_attr):
            build_ops = OpConfig(
                type="reshape2",
                inputs={"X": ["input_data"]},
                outputs={
                    "Out": ["output_data"],
                    "XShape": ["x_shape"],
                },
                attrs={"shape": attr_shape, })
            program_config = ProgramConfig(
                ops=[build_ops],
                weights={},
                inputs={
                    "input_data":
                    TensorConfig(data_gen=partial(generate_input)),
                },
                outputs=["output_data"])
        elif (with_shape_tensor):
            build_ops = OpConfig(
                type="reshape2",
                inputs={"X": ["input_data"],
                        "Shape": ["input_shape"]},
                outputs={
                    "Out": ["output_data"],
                    "XShape": ["x_shape"],
                },
                attrs={})
            program_config = ProgramConfig(
                ops=[build_ops],
                weights={},
                inputs={
                    "input_data":
                    TensorConfig(data_gen=partial(generate_input)),
                    "input_shape":
                    TensorConfig(data_gen=partial(generate_shape)),
                },
                outputs=["output_data"])
        else:
            build_ops = OpConfig(
                type="reshape2",
                inputs={"X": ["input_data"]},
                outputs={
                    "Out": ["output_data"],
                    "XShape": ["x_shape"],
                },
                attrs={})
            program_config = ProgramConfig(
                ops=[build_ops],
                weights={},
                inputs={
                    "input_data":
                    TensorConfig(data_gen=partial(generate_input)),
                },
                outputs=["output_data"])
        return program_config

    def sample_predictor_configs(self):
        atol, rtol = 1e-5, 1e-5
        config_lists = self.get_predictor_configs()
        for config in config_lists:
            if config.target() in [TargetType.Metal]:
                atol, rtol = 2e-4, 2e-4

        return self.get_predictor_configs(), ["reshape2"], (atol, rtol)

    def add_ignore_pass_case(self):
        def teller1(program_config, predictor_config):
            if self.get_nnadapter_device_name() == "nvidia_tensorrt":
                in_shape = program_config.inputs["input_data"].shape
                shape = program_config.ops[0].attrs["shape"]
                if in_shape[0] != shape[0]:
                    return True

        self.add_ignore_check_case(
            teller1, IgnoreReasons.PADDLELITE_NOT_SUPPORT,
            "Lite does not support change batch on nvidia_tensorrt.")

        def _teller2(program_config, predictor_config):
            target_type = predictor_config.target()
            if target_type == TargetType.Metal:
                return True

        self.add_ignore_check_case(_teller2,
                                   IgnoreReasons.PADDLELITE_NOT_SUPPORT,
                                   "Metal report Can't find a io_copy kernel.")

        def teller3(program_config, predictor_config):
            if self.get_nnadapter_device_name() == "intel_openvino":
                return True

        self.add_ignore_check_case(teller3,
                                   IgnoreReasons.PADDLELITE_NOT_SUPPORT,
                                   "intel_openvino report error.")

        def _teller4(program_config, predictor_config):
            target_type = predictor_config.target()
            in_x_shape = list(program_config.inputs["input_data"].shape)
            if target_type not in [
                    TargetType.Metal, TargetType.X86, TargetType.Host,
                    TargetType.ARM, TargetType.OpenCL
            ]:
                if len(in_x_shape) == 0:
                    return True

        self.add_ignore_check_case(
            _teller4, IgnoreReasons.PADDLELITE_NOT_SUPPORT,
            "Only test 0D-tensor on CPU(ARM/Host/Metal/X86/OpenCL) now.")

    def test(self, *args, **kwargs):
        self.run_and_statis(quant=False, max_examples=500)


if __name__ == "__main__":
    unittest.main(argv=[''])
