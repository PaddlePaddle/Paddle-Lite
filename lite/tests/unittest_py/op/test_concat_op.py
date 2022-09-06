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
from functools import partial
import random
import numpy as np
import copy


class TestConcatOp(AutoScanTest):
    def __init__(self, *args, **kwargs):
        AutoScanTest.__init__(self, *args, **kwargs)
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
        self.enable_testing_on_place(
            TargetType.X86,
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
            "cambricon_mlu", "nvidia_tensorrt", "intel_openvino"
        ])

    def is_program_valid(self,
                         program_config: ProgramConfig,
                         predictor_config: CxxConfig) -> bool:
        return True

    def sample_program_configs(self, draw):
        max_num_input = 6
        num_input = draw(st.integers(min_value=2, max_value=max_num_input))
        input_name_list = ["input_data" + str(i) for i in range(num_input)]
        input_type_dict = {"X": input_name_list}

        input_shape0 = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=20), min_size=1, max_size=4))
        axis = draw(st.integers(min_value=-1, max_value=len(input_shape0) - 1))

        # Create n-1 variables: input_shape1, input_shape2, ... , input_shape{n-1}
        for i in range(1, num_input):
            var_name = "input_shape" + str(i)
            globals()[var_name] = copy.deepcopy(input_shape0)

        has_axis_tensor = draw(st.booleans())

        def generate_input(*args, **kwargs):
            i = kwargs["id"]
            input_shape = []
            if i == 0:
                input_shape = input_shape0
            elif i == 1:
                input_shape1[axis] = draw(
                    st.integers(
                        min_value=1, max_value=20))
                input_shape = input_shape1
            elif i == 2:
                input_shape2[axis] = draw(
                    st.integers(
                        min_value=1, max_value=20))
                input_shape = input_shape2
            elif i == 3:
                input_shape3[axis] = draw(
                    st.integers(
                        min_value=1, max_value=20))
                input_shape = input_shape3
            elif i == 4:
                input_shape4[axis] = draw(
                    st.integers(
                        min_value=1, max_value=20))
                input_shape = input_shape4
            elif i == 5:
                input_shape5[axis] = draw(
                    st.integers(
                        min_value=1, max_value=20))
                input_shape = input_shape5
            else:
                print("The max num of inputs is {}, but got input index is {}".
                      format(max_num_input, i))
                sys.exit()
            return np.random.random(input_shape).astype(np.float32)

        def generate_axis(*args, **kwargs):
            return np.array([axis]).astype("int32")

        input_data_dict = {}
        for i in range(num_input):
            input_data_dict[input_name_list[i]] = TensorConfig(
                data_gen=partial(
                    generate_input, id=i))

        if has_axis_tensor:
            input_type_dict["AxisTensor"] = ["axis_tensor_data"]
            input_data_dict["axis_tensor_data"] = TensorConfig(
                data_gen=partial(generate_axis))

        concat_op = OpConfig(
            type="concat",
            inputs=input_type_dict,
            outputs={"Out": ["output_data"]},
            attrs={"axis": axis})
        program_config = ProgramConfig(
            ops=[concat_op],
            weights={},
            inputs=input_data_dict,
            outputs=["output_data"])
        return program_config

    def sample_predictor_configs(self):
        atol, rtol = 1e-5, 1e-5
        target_str = self.get_target()
        if target_str == "Metal":
            atol, rtol = 2e-4, 2e-4
        return self.get_predictor_configs(), ["concat"], (atol, rtol)

    def add_ignore_pass_case(self):
        def _teller1(program_config, predictor_config):
            target_type = predictor_config.target()
            input_shape = program_config.inputs["input_data0"].shape
            if target_type == TargetType.Metal:
                if len(input_shape) != 4:
                    return True

        self.add_ignore_check_case(
            _teller1, IgnoreReasons.PADDLELITE_NOT_SUPPORT,
            "Lite is not supported on metal. We need to fix it as soon as possible."
        )

        def _teller2(program_config, predictor_config):
            if self.get_nnadapter_device_name() == "nvidia_tensorrt":
                in0_shape = program_config.inputs["input_data0"].shape
                axis = program_config.ops[0].attrs["axis"]
                if "axis_tensor_data" in program_config.inputs.keys() \
                    or len(in0_shape) == 1 \
                    or axis == 0:
                    return True

        self.add_ignore_check_case(
            _teller2, IgnoreReasons.PADDLELITE_NOT_SUPPORT,
            "Lite does not support 'AxisTensor input' or 'in_shape_size == 1' or 'axis == 0'."
        )

        def _teller3(program_config, predictor_config):
            if "intel_openvino" in self.get_nnadapter_device_name():
                if "axis_tensor_data" in program_config.inputs.keys():
                    return True

        self.add_ignore_check_case(
            _teller3, IgnoreReasons.PADDLELITE_NOT_SUPPORT,
            "Intel OpenVINO does not support 'AxisTensor input'.")

    def test(self, *args, **kwargs):
        target_str = self.get_target()
        max_examples = 100
        if target_str == "OpenCL":
            # Make sure to generate enough valid cases for OpenCL
            max_examples = 500
        if target_str == "Metal":
            # Make sure to generate enough valid cases for Metal
            max_examples = 400

        self.run_and_statis(
            quant=False, min_success_num=50, max_examples=max_examples)


if __name__ == "__main__":
    unittest.main(argv=[''])
