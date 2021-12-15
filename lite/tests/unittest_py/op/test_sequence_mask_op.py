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
import numpy as np
import hypothesis
from hypothesis import given, settings, seed, example, assume
import hypothesis.strategies as st
import argparse


class TestSequenceMaskOp(AutoScanTest):
    def __init__(self, *args, **kwargs):
        AutoScanTest.__init__(self, *args, **kwargs)
        self.enable_testing_on_place(
            TargetType.Host,
            PrecisionType.FP32,
            DataLayoutType.NCHW,
            thread=[1, 4])

    def is_program_valid(self,
                         program_config: ProgramConfig,
                         predictor_config: CxxConfig) -> bool:
        return True

    def sample_program_configs(self, draw):
        def generate_input(*args, **kwargs):
            if kwargs["type"] == "int32":
                return np.random.randint(kwargs["low"], kwargs["high"],
                                         kwargs["shape"]).astype(np.int32)
            elif kwargs["type"] == "int64":
                return np.random.randint(kwargs["low"], kwargs["high"],
                                         kwargs["shape"]).astype(np.int64)
            elif kwargs["type"] == "float32":
                return kwargs["high"] * np.random.random(kwargs[
                    "shape"]).astype(np.float32) + kwargs["low"]

        input_type = draw(st.sampled_from(["int32", "int64", "float32"]))
        has_max_len_tensor = draw(st.booleans())

        x_shape = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=10), min_size=2, max_size=7))
        maxlen = draw(st.integers(min_value=-10, max_value=10))
        out_dtype_dict = {"2": np.int32, "3": np.int64, "5": np.float32}
        out_dtype = draw(st.sampled_from([2, 3, 5]))
        assume(maxlen != 0 and max(x_shape) <= maxlen)

        input_dict = {"X": ["x_data"]}
        input_data_gen_dict = {
            "x_data": TensorConfig(data_gen=partial(
                generate_input,
                type=input_type,
                low=1,
                high=maxlen,
                shape=x_shape))
        }
        if has_max_len_tensor:
            input_dict["MaxLenTensor"] = ["max_len_tensor"]
            input_data_gen_dict["max_len_tensor"] = TensorConfig(
                data_gen=partial(
                    generate_input,
                    type="int32",
                    low=1,
                    high=2,
                    shape=[maxlen]))

        sequence_mask_op = OpConfig(
            type="sequence_mask",
            inputs=input_dict,
            outputs={"Y": ["output_data"]},
            attrs={"maxlen": maxlen,
                   "out_dtype": out_dtype},
            outputs_dtype={"output_data": out_dtype_dict[str(out_dtype)]})

        program_config = ProgramConfig(
            ops=[sequence_mask_op],
            weights={},
            inputs=input_data_gen_dict,
            outputs=["output_data"])

        return program_config

    def sample_predictor_configs(self):
        return self.get_predictor_configs(), ["sequence_expand_mask"], (1e-5,
                                                                        1e-5)

    def add_ignore_pass_case(self):
        pass

    def test(self, *args, **kwargs):
        self.run_and_statis(quant=False, max_examples=25)


if __name__ == "__main__":
    unittest.main(argv=[''])
