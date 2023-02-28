# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
from hypothesis import given, settings, seed, example, assume, reproduce_failure
import hypothesis.strategies as st
import numpy as np


class TestSplitOp(AutoScanTest):
    def __init__(self, *args, **kwargs):
        AutoScanTest.__init__(self, *args, **kwargs)
        self.enable_testing_on_place(
            TargetType.Host, [PrecisionType.FP32, PrecisionType.INT64],
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
            "kunlunxin_xtcl", "nvidia_tensorrt", "intel_openvino"
        ])

    def is_program_valid(self,
                         program_config: ProgramConfig,
                         predictor_config: CxxConfig) -> bool:
        x_dtype = program_config.inputs["input_data"].dtype
        #check config
        if predictor_config.precision() == PrecisionType.INT64:
            if x_dtype != np.int64:
                return False
        return True

    def sample_program_configs(self, draw):
        in_shape = draw(
            st.sampled_from([[6, 9, 24], [6, 24, 24], [6, 24], [24, 24], [24]
                             ]))
        batch = draw(st.integers(min_value=1, max_value=10))
        in_shape.insert(0, batch)
        sections = draw(
            st.sampled_from([[], [3, 3], [2, 4], [10, 14], [2, 2, 2],
                             [1, 3, 2], [3, 3, 3], [3, 7, 14]]))
        input_num = draw(st.sampled_from([0, 1]))
        num = draw(st.sampled_from([0, 2, 3]))
        input_axis = draw(st.sampled_from([0, 1, 2, 3]))
        input_type = draw(st.sampled_from(["float32", "int32", "int64"]))
        Out = draw(
            st.sampled_from([["output_var0", "output_var1"],
                             ["output_var0", "output_var1", "output_var2"]]))

        #Sections and num cannot both be equal to 0.
        assume((num != 0 and len(sections) == 0) or (num == 0 and
                                                     len(sections) != 0))

        # the dimensions of input and axis match
        assume(input_axis < len(in_shape))

        #When sections and num are not both equal to 0, sections has higher priority.
        #The sum of sections should be equal to the input size.
        if len(sections) != 0:
            assume(len(Out) == len(sections))
            assume(in_shape[input_axis] % len(sections) == 0)
            sum = 0
            for i in sections:
                sum += i
            assume(sum == in_shape[input_axis])

        if num != 0:
            assume(len(Out) == num)
            assume(in_shape[input_axis] % num == 0)

        if input_num == 0:
            assume((len(in_shape) == 2) & (in_shape[1] == 24) & (
                sections == [10, 14]) & (len(Out) == 2))

        def generate_input(*args, **kwargs):
            if input_type == "float32":
                return np.random.normal(0.0, 1.0, in_shape).astype(np.float32)
            elif input_type == "int32":
                return np.random.normal(0.0, 1.0, in_shape).astype(np.int32)
            elif input_type == "int64":
                return np.random.normal(0.0, 1.0, in_shape).astype(np.int64)

        def generate_AxisTensor(*args, **kwargs):
            return np.ones([1]).astype(np.int32)

        def generate_SectionsTensorList1(*args, **kwargs):
            return np.array([10]).astype(np.int32)

        def generate_SectionsTensorList2(*args, **kwargs):
            return np.array([14]).astype(np.int32)

        dics_intput = [{
            "X": ["input_data"],
            "AxisTensor": ["AxisTensor"],
            "SectionsTensorList":
            ["SectionsTensorList1", "SectionsTensorList2"]
        }, {
            "X": ["input_data"]
        }]

        dics_weight = [{
            "AxisTensor": TensorConfig(data_gen=partial(generate_AxisTensor)),
            "SectionsTensorList1":
            TensorConfig(data_gen=partial(generate_SectionsTensorList1)),
            "SectionsTensorList2":
            TensorConfig(data_gen=partial(generate_SectionsTensorList2))
        }, {}]

        ops_config = OpConfig(
            type="split",
            inputs=dics_intput[input_num],
            outputs={"Out": Out},
            attrs={"sections": sections,
                   "num": num,
                   "axis": input_axis})

        program_config = ProgramConfig(
            ops=[ops_config],
            weights=dics_weight[input_num],
            inputs={
                "input_data": TensorConfig(data_gen=partial(generate_input))
            },
            outputs=Out)

        return program_config

    def sample_predictor_configs(self):
        atol, rtol = 1e-5, 1e-5
        config_lists = self.get_predictor_configs()
        for config in config_lists:
            if config.target() in [TargetType.Metal]:
                atol, rtol = 1e-3, 1e-3

        return self.get_predictor_configs(), ["split"], (atol, rtol)

    def add_ignore_pass_case(self):
        def teller1(program_config, predictor_config):
            x_shape = list(program_config.inputs["input_data"].shape)
            if predictor_config.target() == TargetType.Metal:
                if len(x_shape) != 4:
                    return True

        self.add_ignore_check_case(
            teller1, IgnoreReasons.ACCURACY_ERROR,
            "The op output has diff in a specific case. We need to fix it as soon as possible."
        )

        def teller2(program_config, predictor_config):
            x_dtype = program_config.inputs["input_data"].dtype
            x_shape = list(program_config.inputs["input_data"].shape)
            out_shape = list(program_config.outputs)
            axis = program_config.ops[0].attrs["axis"]
            num = program_config.ops[0].attrs["num"]
            if predictor_config.target() == TargetType.OpenCL:
                if num != 2 or x_dtype != np.float32:
                    return True
            if predictor_config.target() == TargetType.Metal:
                if len(x_shape) == 2 or axis == 0 or axis == 1:
                    return True
                if x_dtype != np.float32:
                    return True

        self.add_ignore_check_case(
            teller2, IgnoreReasons.PADDLELITE_NOT_SUPPORT,
            "Lite does not support this op in a specific case. We need to fix it as soon as possible."
        )

        def _teller3(program_config, predictor_config):
            if "nvidia_tensorrt" in self.get_nnadapter_device_name():
                in_shape = program_config.inputs["input_data"].shape
                axis = program_config.ops[0].attrs["axis"]
                in_dtype = program_config.inputs["input_data"].dtype
                if len(in_shape) == 1 or axis == 0 or in_dtype != np.float32:
                    return True

        self.add_ignore_check_case(
            _teller3, IgnoreReasons.PADDLELITE_NOT_SUPPORT,
            "Lite does not support 'in_shape_size == 1' or 'axis == 0' or 'in_dtype != float32' on NvidiaTensorrt."
        )

    def test(self, *args, **kwargs):
        target_str = self.get_target()
        max_examples = 50
        if target_str == "OpenCL":
            # Make sure to generate enough valid cases for OpenCL
            max_examples = 100
        if target_str == "Metal":
            # Make sure to generate enough valid cases for OpenCL
            max_examples = 500
        if self.get_nnadapter_device_name() == "kunlunxin_xtcl":
            max_examples = 500

        self.run_and_statis(
            quant=False, min_success_num=25, max_examples=max_examples)


if __name__ == "__main__":
    unittest.main(argv=[''])
