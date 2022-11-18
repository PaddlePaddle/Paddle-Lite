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
import hypothesis
from hypothesis import given, settings, seed, example, assume
import hypothesis.strategies as st
import numpy as np


class TestSetValueOp(AutoScanTest):
    def __init__(self, *args, **kwargs):
        AutoScanTest.__init__(self, *args, **kwargs)
        self.enable_testing_on_place(
            TargetType.X86, [PrecisionType.FP32],
            DataLayoutType.NCHW,
            thread=[1, 4])
        self.enable_testing_on_place(
            TargetType.Host, [PrecisionType.FP32],
            DataLayoutType.NCHW,
            thread=[1, 4])

    def is_program_valid(self,
                         program_config: ProgramConfig,
                         predictor_config: CxxConfig) -> bool:
        target_type = predictor_config.target()
        input_data_type = program_config.inputs["input_data"].dtype
        # Check config
        if target_type in [TargetType.X86]:
            if predictor_config.precision(
            ) == PrecisionType.FP32 and input_data_type != np.float32:
                return False
        return True

    def sample_program_configs(self, draw):
        in_shape = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=100),
                min_size=1,
                max_size=1))

        input_type = draw(st.sampled_from(["float32"]))
        value_tensor_flag = True

        def generate_input(*args, **kwargs):
            if input_type == "float32":
                return np.random.normal(1.0, 6.0, in_shape).astype(np.float32)

        inputs_dict = {'Input': ['input_data']}
        if value_tensor_flag:
            inputs_dict['ValueTensor'] = ['value_data']
        ops_config = OpConfig(
            type="set_value",
            inputs=inputs_dict,
            outputs={"Out": ["output_data"]},
            attrs={
                'shape': in_shape,
                'axes': [0],
                'starts': [0],
                'ends': [in_shape[0]],
                'steps': [1],
                'decrease_axes': [],
                'none_axes': [],
                'fp32_values': []
            })

        inputs_data_dict = {
            "input_data": TensorConfig(data_gen=partial(generate_input))
        }
        if value_tensor_flag:
            inputs_data_dict['value_data'] = TensorConfig(
                data_gen=partial(generate_input))

        program_config = ProgramConfig(
            ops=[ops_config],
            weights={},
            inputs=inputs_data_dict,
            outputs=["output_data"])

        return program_config

    def sample_predictor_configs(self):
        return self.get_predictor_configs(), ["set_value"], (1e-5, 1e-5)

    def add_ignore_pass_case(self):
        pass

    def test(self, *args, **kwargs):
        self.run_and_statis(quant=False, max_examples=25)


if __name__ == "__main__":
    unittest.main(argv=[''])
