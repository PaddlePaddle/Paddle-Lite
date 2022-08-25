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


class TestFlattenContiguousRangeOp(AutoScanTest):
    def __init__(self, *args, **kwargs):
        AutoScanTest.__init__(self, *args, **kwargs)
        self.enable_testing_on_place(
            TargetType.Host,
            PrecisionType.Any,
            DataLayoutType.Any,
            thread=[1, 4])
        self.enable_testing_on_place(TargetType.NNAdapter, PrecisionType.FP32)
        self.enable_devices_on_nnadapter(device_names=[
            "cambricon_mlu", "nvidia_tensorrt", "intel_openvino"
        ])

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

        start_axis = draw(
            st.integers(
                min_value=0, max_value=len(in_shape) - 1))
        stop_axis = draw(
            st.integers(
                min_value=start_axis, max_value=len(in_shape) - 1))

        outputs_ = ["output_data", "xshape_data"]
        target_str = self.get_target()
        if target_str == "NNAdapter":
            if "nvidia_tensorrt" in self.get_nnadapter_device_name():
                assume(input_type != "int64")

        if self.get_nnadapter_device_name(
        ) in ["nvidia_tensorrt", "intel_openvino"]:
            outputs_ = ["output_data"]

        flatten_contiguous_range_op = OpConfig(
            type="flatten_contiguous_range",
            inputs={"X": ["input_data"]},
            outputs={"Out": ["output_data"],
                     "XShape": ["xshape_data"]},
            attrs={"start_axis": start_axis,
                   "stop_axis": stop_axis})
        flatten_contiguous_range_op.outputs_dtype = {"output_data": input_type}
        program_config = ProgramConfig(
            ops=[flatten_contiguous_range_op],
            weights={"xshape_data": TensorConfig(shape=in_shape)},
            inputs={
                "input_data": TensorConfig(data_gen=partial(
                    generate_input,
                    type=input_type,
                    low=-10,
                    high=10,
                    shape=in_shape))
            },
            outputs=outputs_)

        return program_config

    def sample_predictor_configs(self):
        return self.get_predictor_configs(), ["flatten_contiguous_range"], (
            1e-5, 1e-5)

    def add_ignore_pass_case(self):
        def teller1(program_config, predictor_config):
            if "nvidia_tensorrt" in self.get_nnadapter_device_name():
                in_shape = program_config.inputs["input_data"].shape
                start_axis = program_config.ops[0].attrs["start_axis"]
                if len(in_shape) == 1 or start_axis == 0:
                    return True

        self.add_ignore_check_case(
            teller1, IgnoreReasons.PADDLELITE_NOT_SUPPORT,
            "Lite does not support 'in_shape_size == 1' or 'start_axis == 0' on NvidiaTensorrt."
        )

    def test(self, *args, **kwargs):
        self.run_and_statis(quant=False, max_examples=300)


if __name__ == "__main__":
    unittest.main(argv=[''])
