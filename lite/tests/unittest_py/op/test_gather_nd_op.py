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


class TestGatherNdOp(AutoScanTest):
    def __init__(self, *args, **kwargs):
        AutoScanTest.__init__(self, *args, **kwargs)
        self.enable_testing_on_place(
            TargetType.Host,
            PrecisionType.Any,
            DataLayoutType.Any,
            thread=[1, 4])

    def is_program_valid(self,
                         program_config: ProgramConfig,
                         predictor_config: CxxConfig) -> bool:
        return True

    def sample_program_configs(self, draw):
        in_shape = draw(
            st.lists(
                st.integers(
                    min_value=4, max_value=8), min_size=3, max_size=4))
        value0 = draw(st.integers(min_value=0, max_value=in_shape[0] - 1))
        value1 = draw(st.integers(min_value=0, max_value=in_shape[1] - 1))
        value2 = draw(st.integers(min_value=0, max_value=in_shape[2] - 1))
        index = draw(
            st.sampled_from([[value0], [value0, value1],
                             [value0, value1, value2]]))
        index_type = draw(st.sampled_from(["int32", "int64"]))

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

        input_type = draw(st.sampled_from(["float32", "int64", "int32"]))

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

        gather_nd_op = OpConfig(
            type="gather_nd",
            inputs=op_inputs,
            outputs={"Out": ["output_data"]},
            attrs={"axis": 1})
        program_config = ProgramConfig(
            ops=[gather_nd_op],
            weights={},
            inputs=program_inputs,
            outputs=["output_data"])
        return program_config

    def sample_predictor_configs(self):
        return self.get_predictor_configs(), ["gather_nd"], (1e-5, 1e-5)

    def add_ignore_pass_case(self):
        def _teller1(program_config, predictor_config):
            if predictor_config.target() == TargetType.Host:
                in_dtype = program_config.inputs["input_data"].dtype
                #wait for atuo_scan_base bug fix 
                if "float32" != in_dtype:
                    return True

        self.add_ignore_check_case(
            _teller1, IgnoreReasons.PADDLELITE_NOT_SUPPORT,
            "Lite does not support this op in a specific case on host. We need to fix it as soon as possible."
        )

    def test(self, *args, **kwargs):
        self.run_and_statis(quant=False, max_examples=300)


if __name__ == "__main__":
    unittest.main(argv=[''])
