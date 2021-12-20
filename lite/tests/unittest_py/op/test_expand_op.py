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


class TestExpandOp(AutoScanTest):
    def __init__(self, *args, **kwargs):
        AutoScanTest.__init__(self, *args, **kwargs)
        host_places = [
            Place(TargetType.Host, PrecisionType.INT32, DataLayoutType.NCHW),
            Place(TargetType.Host, PrecisionType.INT64, DataLayoutType.NCHW),
            Place(TargetType.Host, PrecisionType.FP32, DataLayoutType.NCHW)
        ]
        self.enable_testing_on_place(thread=[1, 4], places=host_places)

    def is_program_valid(self,
                         program_config: ProgramConfig,
                         predictor_config: CxxConfig) -> bool:
        in_dtype = program_config.inputs["input_data"].dtype
        #wait for atuo_scan_base bug fix 
        if "float32" != in_dtype:
            return False
        return True

    def sample_program_configs(self, draw):
        in_shape = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=8), min_size=3, max_size=4))
        expand_shape = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=4),
                min_size=len(in_shape),
                max_size=len(in_shape)))
        with_tensor = draw(st.sampled_from([True, False]))

        def generate_shape(*args, **kwargs):
            return np.array(expand_shape).astype(np.int32)

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

        input_type = draw(st.sampled_from(["float32", "int32", "int64"]))

        def gnerate_inputs(with_tensor):
            inputs1 = {}
            inputs2 = {}
            if (with_tensor):
                inputs1 = {"X": ["input_data"], "ExpandTimes": ["expand_data"]}
                inputs2 = {
                    "input_data": TensorConfig(data_gen=partial(
                        generate_input,
                        type=input_type,
                        low=-10,
                        high=10,
                        shape=in_shape)),
                    "expand_data":
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
            st.lists(
                st.integers(
                    min_value=1, max_value=8),
                min_size=len(in_shape),
                max_size=len(in_shape)))
        inputs = gnerate_inputs(with_tensor)
        expand_op = OpConfig(
            type="expand",
            inputs=inputs[0],
            outputs={"Out": ["output_data"]},
            attrs={"expand_times": attr_shape})

        program_config = ProgramConfig(
            ops=[expand_op],
            weights={},
            inputs=inputs[1],
            outputs=["output_data"])
        return program_config

    def sample_predictor_configs(self):
        return self.get_predictor_configs(), ["expand"], (1e-5, 1e-5)

    def add_ignore_pass_case(self):
        pass

    def test(self, *args, **kwargs):
        self.run_and_statis(quant=False, max_examples=300)


if __name__ == "__main__":
    unittest.main(argv=[''])
