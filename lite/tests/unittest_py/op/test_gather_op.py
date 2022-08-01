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


class TestGatherOp(AutoScanTest):
    def __init__(self, *args, **kwargs):
        AutoScanTest.__init__(self, *args, **kwargs)
        self.enable_testing_on_place(
            TargetType.Host,
            PrecisionType.FP32,
            DataLayoutType.NCHW,
            thread=[1, 2])
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
                    min_value=4, max_value=4), min_size=2, max_size=2))
        axis = draw(st.integers(min_value=0, max_value=len(in_shape) - 1))
        index = draw(
            st.sampled_from([[0], [2], [3], [1, 2], [1, 2, 3], [
                in_shape[axis] - 1
            ], [in_shape[axis] - 2, in_shape[axis] - 1]]))
        axis_type = draw(st.sampled_from(["int32", "int64"]))
        index_type = draw(st.sampled_from(["int32", "int64"]))
        with_tenor_axis = draw(st.booleans())
        input_type = draw(st.sampled_from(["float32", "int64", "int32"]))
        if "intel_openvino" in self.get_nnadapter_device_name():
            with_tenor_axis = False
        if self.get_target() == "OpenCL":
            axis_type = "int32"
            index_type = "int32"
            input_type = "float32"
            with_tenor_axis = True

        def generate_axis(*args, **kwargs):
            if axis_type == "int32":
                return np.array([axis]).astype(np.int32)
            else:
                return np.array([axis]).astype(np.int64)

        def generate_index(*args, **kwargs):
            if index_type == "int32":
                return np.array(index).astype(np.int32)
            else:
                return np.array(index).astype(np.int64)

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

        op_inputs = {}
        program_inputs = {}
        if (with_tenor_axis):
            op_inputs = {
                "X": ["input_data"],
                "Index": ["index_data"],
                "Axis": ["axis_data"]
            }
            program_inputs = {
                "input_data": TensorConfig(data_gen=partial(
                    generate_input,
                    type=input_type,
                    low=-10,
                    high=10,
                    shape=in_shape)),
                "index_data": TensorConfig(data_gen=partial(generate_index)),
                "axis_data": TensorConfig(data_gen=partial(generate_axis))
            }
        else:
            op_inputs = {"X": ["input_data"], "Index": ["index_data"]}
            program_inputs = {
                "input_data": TensorConfig(data_gen=partial(
                    generate_input,
                    type=input_type,
                    low=-10,
                    high=10,
                    shape=in_shape)),
                "index_data": TensorConfig(data_gen=partial(generate_index))
            }
        gather_op = OpConfig(
            type="gather",
            inputs=op_inputs,
            outputs={"Out": ["output_data"]},
            attrs={"axis": axis})
        gather_op.outputs_dtype = {"output_data": input_type}
        program_config = ProgramConfig(
            ops=[gather_op],
            weights={},
            inputs=program_inputs,
            outputs=["output_data"])
        return program_config

    def sample_predictor_configs(self):
        return self.get_predictor_configs(), ["gather"], (1e-5, 1e-5)

    def add_ignore_pass_case(self):
        def _teller1(program_config, predictor_config):
            in_dtype = program_config.inputs["input_data"].dtype
            index_dtpye = program_config.inputs["index_data"].dtype
            in_shape = list(program_config.inputs["input_data"].shape)
            if predictor_config.target() == TargetType.OpenCL:
                axis_dtpye = program_config.inputs["axis_data"].dtype
                if "int32" != axis_dtpye or "int32" != index_dtpye or len(
                        in_shape) != 2:
                    return True

        self.add_ignore_check_case(
            _teller1, IgnoreReasons.ACCURACY_ERROR,
            "The op output has diff in a specific case on opencl. We need to fix it as soon as possible."
        )

    def test(self, *args, **kwargs):
        target_str = self.get_target()
        max_examples = 200
        if target_str == "OpenCL":
            # Make sure to generate enough valid cases for OpenCL
            max_examples = 1000
        self.run_and_statis(quant=False, max_examples=max_examples)


if __name__ == "__main__":
    unittest.main(argv=[''])
