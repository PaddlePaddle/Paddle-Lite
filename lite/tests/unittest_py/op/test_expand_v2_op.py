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


class TestExpandV2Op(AutoScanTest):
    def __init__(self, *args, **kwargs):
        AutoScanTest.__init__(self, *args, **kwargs)
        self.enable_testing_on_place(
            TargetType.Host,
            PrecisionType.FP32,
            DataLayoutType.Any,
            thread=[1, 4])
        self.enable_testing_on_place(TargetType.NNAdapter, PrecisionType.FP32)
        self.enable_devices_on_nnadapter(
            device_names=["cambricon_mlu", "intel_openvino"])

    def is_program_valid(self,
                         program_config: ProgramConfig,
                         predictor_config: CxxConfig) -> bool:
        return True

    def sample_program_configs(self, draw):
        in_shape = draw(st.sampled_from([[1, 1, 1], [2, 1, 4]]))
        Shape = draw(st.sampled_from([[2, 4, 4], [3, 2, 3, 4]]))
        expand_shape = draw(st.sampled_from([[2, 5, 4], [2, 3, 4]]))
        with_Shape = draw(st.sampled_from([True, False]))

        #todo daming5432 input vector tensor
        with_expand_shape = draw(st.booleans())

        if "intel_openvino" in self.get_nnadapter_device_name():
            with_Shape = False
            with_expand_shape = False

        def generate_shape(*args, **kwargs):
            return np.array(Shape).astype(np.int32)

        def generate_expand_shape(i, *args, **kwargs):
            return np.array([expand_shape[i]]).astype(np.int32)

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

        def gnerate_inputs(with_Shape, with_expand_shape):
            inputs1 = {}
            inputs2 = {}
            # with_Shape has a higher priority than expand_shapes_tensor
            if (with_Shape):
                inputs1 = {
                    "X": ["input_data"],
                    "Shape": ["shape_data"],
                }
                inputs2 = {
                    "input_data": TensorConfig(data_gen=partial(
                        generate_input,
                        type=input_type,
                        low=-10,
                        high=10,
                        shape=in_shape)),
                    "shape_data":
                    TensorConfig(data_gen=partial(generate_shape))
                }
            elif ((not with_Shape) and with_expand_shape):
                inputs1 = {
                    "X": ["input_data"],
                    "expand_shapes_tensor":
                    ["expand_data0", "expand_data1", "expand_data2"]
                }
                inputs2 = {
                    "input_data": TensorConfig(data_gen=partial(
                        generate_input,
                        type=input_type,
                        low=-10,
                        high=10,
                        shape=in_shape)),
                    "expand_data0":
                    TensorConfig(data_gen=partial(generate_expand_shape, 0)),
                    "expand_data1":
                    TensorConfig(data_gen=partial(generate_expand_shape, 1)),
                    "expand_data2":
                    TensorConfig(data_gen=partial(generate_expand_shape, 2))
                }
            elif (with_Shape and (not with_expand_shape)):
                inputs1 = {"X": ["input_data"], "Shape": ["shape_data"]}
                inputs2 = {
                    "input_data": TensorConfig(data_gen=partial(
                        generate_input,
                        type=input_type,
                        low=-10,
                        high=10,
                        shape=in_shape)),
                    "shape_data":
                    TensorConfig(data_gen=partial(generate_shape))
                }
            else:
                inputs1 = {"X": ["input_data"]}
                inputs2 = {
                    "input_data": TensorConfig(data_gen=partial(
                        generate_input,
                        type=input_type,
                        low=-10,
                        high=10,
                        shape=in_shape))
                }
            return [inputs1, inputs2]

        attr_shape = draw(
            st.sampled_from([[2, 3, 4], [2, 4, 4], [2, 2, 3, 4], [3, 2, 5, 4]
                             ]))
        inputs = gnerate_inputs(with_Shape, with_expand_shape)
        expand_v2_op = OpConfig(
            type="expand_v2",
            inputs=inputs[0],
            outputs={"Out": ["output_data"]},
            attrs={"shape": attr_shape})
        expand_v2_op.outputs_dtype = {"output_data": input_type}

        program_config = ProgramConfig(
            ops=[expand_v2_op],
            weights={},
            inputs=inputs[1],
            outputs=["output_data"])
        return program_config

    def sample_predictor_configs(self):
        return self.get_predictor_configs(), ["expand_v2"], (1e-5, 1e-5)

    def add_ignore_pass_case(self):
        pass

    def test(self, *args, **kwargs):
        self.run_and_statis(quant=False, max_examples=300)


if __name__ == "__main__":
    unittest.main(argv=[''])
