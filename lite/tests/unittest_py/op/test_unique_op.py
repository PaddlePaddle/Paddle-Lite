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
from functools import partial
import random
import numpy as np


class TestUniqueOp(AutoScanTest):
    def __init__(self, *args, **kwargs):
        AutoScanTest.__init__(self, *args, **kwargs)
        host_places = [
            Place(TargetType.Host, PrecisionType.FP32, DataLayoutType.NCHW)
        ]
        self.enable_testing_on_place(places=host_places, thread=[1,4])

    def is_program_valid(self,
                         program_config: ProgramConfig,
                         predictor_config: CxxConfig) -> bool:
        return True

    def sample_program_configs(self, draw):
        in_shape = draw(
            st.lists(
                st.integers(
                    min_value=2, max_value=100),
                min_size=1,
                max_size=1))
        in_dtype = draw(st.sampled_from([np.float32, np.int32, np.int64]))
                
        def generate_X_data():
            return np.random.normal(0.0, 5.0, in_shape).astype(in_dtype)

        def generate_IndexTensor():
            return np.random.randint(1, 5, size=in_shape).astype(np.int32)

        dtype = 2
        is_sorted = draw(st.sampled_from([True, False]))
        return_index = draw(st.sampled_from([False]))
        if is_sorted: 
            return_inverse = draw(st.sampled_from([True, False]))
        else:
            return_inverse = True
        return_counts = draw(st.sampled_from([False]))
        outputs = [
            "Out_data"
        ]
        outputs_config = {
            "Out": ["Out_data"]
        }
        outputs_dtype = {
            "Out_data": in_dtype
        }
        if return_inverse:
            outputs.append("Index_data")
            outputs_config["Index"] = ["Index_data"]
            outputs_dtype["Index_data"] = np.int32
        if return_index:
            outputs.append("Indices_data")
            outputs_config["Indices"] = ["Indices_data"]
            outputs_dtype["Indices_data"] = np.int32
        if return_counts:
            outputs.append("Counts_data")
            outputs_config["Counts"] = ["Counts_data"] 
            outputs_dtype["Counts_data"] = np.int32

        axis = draw(st.sampled_from([[0, 1, 2], [1], [0, 2], [2, 1], [0, 1]]))
        axis = []        

        unique_op = OpConfig(
            type = "unique",
            inputs = {"X": ["input_data"]},
            outputs = outputs_config,
            attrs={
                "dtype": 2,
                "return_index": return_index,
                "return_inverse": return_inverse,
                "return_counts": return_counts,
                "axis": axis,
                "is_sorted": is_sorted
            }
        )

        unique_op.outputs_dtype = outputs_dtype

        program_config = ProgramConfig(
            ops=[unique_op],
            weights={},
            inputs={
                "input_data": TensorConfig(data_gen=partial(generate_X_data))
            },
            outputs=outputs
        )
        return program_config

    def sample_predictor_configs(self):
        return self.get_predictor_configs(), [""], (1e-5, 1e-5)

    def add_ignore_pass_case(self):
        pass

    def test(self, *args, **kwargs):
        self.run_and_statis(quant=False, max_examples=25)

if __name__ == "__main__":
    unittest.main(argv=[''])    
