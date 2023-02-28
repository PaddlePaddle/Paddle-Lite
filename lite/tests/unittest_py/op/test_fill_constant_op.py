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

import numpy as np
from functools import partial
import hypothesis.strategies as st


class TestFillConstantOp(AutoScanTest):
    def __init__(self, *args, **kwargs):
        AutoScanTest.__init__(self, *args, **kwargs)
        self.enable_testing_on_place(
            TargetType.Host,
            PrecisionType.FP32,
            DataLayoutType.NCHW,
            thread=[1, 2, 4])
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
        tensor_shape = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=10), min_size=1, max_size=4))
        dtype = draw(st.sampled_from([2, 3, 5]))

        with_value_tensor = draw(st.sampled_from([True, False]))
        with_shape_tensor = draw(st.sampled_from([True, False]))
        if "nvidia_tensorrt" in self.get_nnadapter_device_name(
        ) or "intel_openvino" in self.get_nnadapter_device_name():
            with_shape_tensor = False
        # nvidia_tensorrt now just supports shape is from attr

        def generate_shape_tensor(*args, **kwargs):
            return np.array(tensor_shape).astype(np.int32)

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

        if dtype == 2:
            input_type = "int32"
        elif dtype == 3:
            input_type = "int64"
        else:
            input_type = "float32"

        value = draw(st.floats(min_value=-10, max_value=10))
        op_inputs = {}
        program_inputs = {}

        #ShapeTensorList not support now 
        if (with_value_tensor and with_shape_tensor):
            op_inputs = {
                "ValueTensor": ["value_data"],
                "ShapeTensor": ["shape_data"]
            }
            program_inputs = {
                "value_data": TensorConfig(data_gen=partial(
                    generate_input,
                    type=input_type,
                    low=-10,
                    high=10,
                    shape=[1])),
                "shape_data":
                TensorConfig(data_gen=partial(generate_shape_tensor))
            }
        elif ((not with_value_tensor) and with_shape_tensor):
            op_inputs = {"ShapeTensor": ["shape_data"]}
            program_inputs = {
                "shape_data":
                TensorConfig(data_gen=partial(generate_shape_tensor))
            }
        elif (with_value_tensor and (not with_shape_tensor)):
            op_inputs = {"ValueTensor": ["value_data"]}
            program_inputs = {
                "value_data": TensorConfig(data_gen=partial(
                    generate_input,
                    type=input_type,
                    low=-10,
                    high=10,
                    shape=[1]))
            }

        fill_constant_op = OpConfig(
            type="fill_constant",
            inputs=op_inputs,
            outputs={"Out": ["output_data"]},
            attrs={
                "dtype": dtype,
                "shape": in_shape,
                "value": value,
                "force_cpu": False
            })
        program_config = ProgramConfig(
            ops=[fill_constant_op],
            weights={},
            inputs=program_inputs,
            outputs=["output_data"])
        return program_config

    def sample_predictor_configs(self):
        return self.get_predictor_configs(), ["fill_constant"], (1e-5, 1e-5)

    def add_ignore_pass_case(self):
        def teller1(program_config, predictor_config):
            if self.get_nnadapter_device_name() == "nvidia_tensorrt":
                dtype = program_config.ops[0].attrs["dtype"]
                in_num = len(program_config.inputs)
                if dtype != 5 or in_num != 0:
                    return True

        self.add_ignore_check_case(
            teller1, IgnoreReasons.PADDLELITE_NOT_SUPPORT,
            "Lite does not support 'shape is form tensor' or 'value is from tensor' "
            "or 'dtype is not float32' on nvidia_tensorrt.")

        def teller2(program_config, predictor_config):
            if self.get_nnadapter_device_name() == "kunlunxin_xtcl":
                in_num = len(program_config.inputs)
                if in_num != 0:
                    return True

        self.add_ignore_check_case(
            teller2, IgnoreReasons.PADDLELITE_NOT_SUPPORT,
            "Lite does not support 'shape is form tensor' or 'value is from tensor' on kunlunxin_xtcl."
        )

    def test(self, *args, **kwargs):
        self.run_and_statis(quant=False, max_examples=25)


if __name__ == "__main__":
    unittest.main(argv=[''])
