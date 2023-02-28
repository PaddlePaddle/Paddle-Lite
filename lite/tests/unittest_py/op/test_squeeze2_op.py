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
from functools import partial
import hypothesis
from hypothesis import given, settings, seed, example, assume
import hypothesis.strategies as st
import numpy as np


class TestSqueeze2Op(AutoScanTest):
    def __init__(self, *args, **kwargs):
        AutoScanTest.__init__(self, *args, **kwargs)
        self.enable_testing_on_place(
            TargetType.Host,
            PrecisionType.Any,
            DataLayoutType.NCHW,
            thread=[1, 4])
        opencl_places = [
            Place(TargetType.OpenCL, PrecisionType.FP32, DataLayoutType.NCHW),
            Place(TargetType.OpenCL, PrecisionType.FP16,
                  DataLayoutType.ImageDefault), Place(
                      TargetType.OpenCL, PrecisionType.FP16,
                      DataLayoutType.ImageFolder), Place(
                          TargetType.OpenCL, PrecisionType.Any,
                          DataLayoutType.ImageDefault), Place(
                              TargetType.OpenCL, PrecisionType.Any,
                              DataLayoutType.ImageFolder),
            Place(TargetType.OpenCL, PrecisionType.Any, DataLayoutType.NCHW),
            Place(TargetType.Host, PrecisionType.FP32)
        ]
        self.enable_testing_on_place(places=opencl_places)
        self.enable_testing_on_place(TargetType.NNAdapter, PrecisionType.FP32)
        self.enable_devices_on_nnadapter(device_names=[
            "kunlunxin_xtcl", "cambricon_mlu", "nvidia_tensorrt",
            "intel_openvino"
        ])

    def is_program_valid(self,
                         program_config: ProgramConfig,
                         predictor_config: CxxConfig) -> bool:
        #check config
        x_dtype = program_config.inputs["input_data"].dtype
        if predictor_config.target() == TargetType.OpenCL:
            if x_dtype == np.int32 or x_dtype == np.int64:
                return False
        return True

    def sample_program_configs(self, draw):
        in_shape = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=32), min_size=1, max_size=4))
        input_type = draw(st.sampled_from(["int64", "int32", "float32"]))
        input_axis = draw(
            st.sampled_from([[0, 1, 2, 3], [-1, 2, 3], [], [-1], [1], [2],
                             [3], [-1, 0, 1]]))
        has_xshape = draw(st.sampled_from([True, False]))
        assume(len(input_axis) <= len(in_shape))
        if len(input_axis) > 0:
            for num in input_axis:
                num = num if num >= 0 else num + len(in_shape)
                assume(num < len(in_shape))
        # "nvidia_tensorrt" must satisfies theses 
        if self.get_nnadapter_device_name() == "nvidia_tensorrt":
            for i in range(len(input_axis)):
                in_shape[input_axis[i]] = 1
            input_type = "float32"

        def generate_input(*args, **kwargs):
            if input_type == "float32":
                return np.random.normal(1.0, 6.0, in_shape).astype(np.float32)
            elif input_type == "int32":
                return np.random.normal(1.0, 6.0, in_shape).astype(np.int32)
            elif input_type == "int64":
                return np.random.normal(1.0, 6.0, in_shape).astype(np.int64)

        def generate_xshape(*args, **kwargs):
            return np.random.normal(1.0, 1.0, in_shape).astype(np.float32)

        if has_xshape == True:
            ops_config = OpConfig(
                type="squeeze2",
                inputs={"X": ["input_data"]},
                outputs={
                    "Out": ["output_data"],
                    "XShape": ["squeeze2_xshape"]
                },
                attrs={"axes": input_axis})

            ops_config.outputs_dtype = {"output_data": input_type}

            program_config = ProgramConfig(
                ops=[ops_config],
                weights={
                    "squeeze2_xshape":
                    TensorConfig(data_gen=partial(generate_xshape))
                },
                inputs={
                    "input_data":
                    TensorConfig(data_gen=partial(generate_input))
                },
                outputs=["output_data"])

        if has_xshape == False:
            ops_config = OpConfig(
                type="squeeze2",
                inputs={"X": ["input_data"]},
                outputs={"Out": ["output_data"]},
                attrs={"axes": input_axis})

            ops_config.outputs_dtype = {"output_data": input_type}

            program_config = ProgramConfig(
                ops=[ops_config],
                weights={},
                inputs={
                    "input_data":
                    TensorConfig(data_gen=partial(generate_input))
                },
                outputs=["output_data"])

        return program_config

    def sample_predictor_configs(self):
        return self.get_predictor_configs(), ["squeeze2"], (1e-5, 1e-5)

    def add_ignore_pass_case(self):
        def teller1(program_config, predictor_config):
            if self.get_nnadapter_device_name() == "nvidia_tensorrt":
                in_shape = program_config.inputs["input_data"].shape
                axes = program_config.ops[0].attrs["axes"]
                if len(in_shape) == 1 \
                    or 0 in axes \
                    or -len(in_shape) in axes \
                    or (in_shape[0] == 1 and len(axes)==0):
                    return True

        self.add_ignore_check_case(
            teller1, IgnoreReasons.PADDLELITE_NOT_SUPPORT,
            "Lite does not support 'in_shape_size == 1' or 'axes has 0' on nvidia_tensorrt."
        )

        def teller2(program_config, predictor_config):
            if self.get_nnadapter_device_name() is not None:
                in_shape = program_config.inputs["input_data"].shape
                axes = program_config.ops[0].attrs["axes"]
                for i in axes:
                    if in_shape[i] != 1:
                        return True

        self.add_ignore_check_case(
            teller2, IgnoreReasons.PADDLELITE_NOT_SUPPORT,
            "Lite does not support 'the axis that will be squeezed is not one' on nnadapter."
        )

        def teller3(program_config, predictor_config):
            if self.get_nnadapter_device_name() == "kunlunxin_xtcl":
                in_shape = program_config.inputs["input_data"].shape
                axes = program_config.ops[0].attrs["axes"]
                if len(axes) == 0:
                    return True
                if len(axes) == len(in_shape):
                    is_all_one = True
                    for i in axes:
                        if in_shape[i] != 1:
                            is_all_one = False
                    if is_all_one:
                        return True

        self.add_ignore_check_case(
            teller3, IgnoreReasons.PADDLELITE_NOT_SUPPORT,
            "Lite does not support 'all axis of inputs are squeezed' on kunlunxin_xtcl. because output shape is wrong"
        )

    def test(self, *args, **kwargs):
        target_str = self.get_target()
        max_examples = 200
        if target_str == "NNAdapter":
            max_examples = 2000
        self.run_and_statis(quant=False, max_examples=max_examples)


if __name__ == "__main__":
    unittest.main(argv=[''])
