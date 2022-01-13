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
from hypothesis import given, settings, seed, example, assume, reproduce_failure
import hypothesis.strategies as st
import numpy as np
from functools import partial


class TestCtcAlignOp(AutoScanTest):
    def __init__(self, *args, **kwargs):
        AutoScanTest.__init__(self, *args, **kwargs)
        self.enable_testing_on_place(
            TargetType.Host, [PrecisionType.INT32, PrecisionType.INT64],
            DataLayoutType.NCHW,
            thread=[1, 4])

    def is_program_valid(self,
                         program_config: ProgramConfig,
                         predictor_config: CxxConfig) -> bool:
        target_type = predictor_config.target()
        input_data_type = program_config.inputs["input_data"].dtype
        # Check config
        if predictor_config.precision(
        ) == PrecisionType.INT64 and input_data_type != np.int64:
            return False
        if predictor_config.precision(
        ) == PrecisionType.INT32 and input_data_type != np.int32:
            return False
        return True

    def sample_program_configs(self, draw):
        blank = draw(st.integers(min_value=0, max_value=10))
        merge_repeated = draw(st.booleans())
        padding_value = draw(st.integers(min_value=0, max_value=10))
        is_input_lod_tensor = draw(st.booleans())
        input_output_data_type = draw(st.sampled_from([np.int32, np.int64]))
        # LoD tensor need
        sequence_num = draw(st.integers(min_value=1, max_value=20))
        max_sequence_length = draw(st.integers(min_value=10, max_value=20))

        def gen_lod_info_data(sequence_num, max_sequence_length):
            lod_data_init_list = [
                np.random.randint(1, max_sequence_length + 1)
                for i in range(sequence_num)
            ]
            lod_data_init_list.insert(0, 0)
            lod_data_init_array = np.array(lod_data_init_list)
            lod_info_data = np.cumsum(lod_data_init_array)
            return [lod_info_data.tolist()]

        if is_input_lod_tensor:
            input_lod_data = gen_lod_info_data(sequence_num,
                                               max_sequence_length)
        else:
            input_lod_data = None

        input_2d_tensor_shape = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=6), min_size=2, max_size=2))

        def gen_input_data():
            if is_input_lod_tensor:
                total_sequence_length_of_mini_batch = input_lod_data[-1][-1]
                input_data = np.random.randint(
                    0,
                    10,
                    size=(total_sequence_length_of_mini_batch, 1),
                    dtype=input_output_data_type)
                return input_data
            else:
                return np.random.randint(
                    0,
                    10,
                    size=input_2d_tensor_shape,
                    dtype=input_output_data_type)

        def gen_input_length():
            input_length_data = []
            for i in range(input_2d_tensor_shape[0]):
                random_data = np.random.randint(
                    0, input_2d_tensor_shape[1], dtype=input_output_data_type)
                input_length_data.append(random_data)
            return np.array(input_length_data).reshape(-1, 1)

        def gen_op_inputs_outputs_attrs():
            #inputs
            inputs = {'Input': ['input_data']}
            inputs_tensor = {
                'input_data': TensorConfig(
                    data_gen=partial(gen_input_data), lod=input_lod_data)
            }
            if not is_input_lod_tensor:
                inputs['InputLength'] = ["input_length_data"]
                inputs_tensor['input_length_data'] = TensorConfig(
                    data_gen=partial(gen_input_length))
            # attrs
            attrs = {
                'blank': blank,
                'merge_repeated': merge_repeated,
                'padding_value': padding_value
            }
            # output
            outputs_tensor_name_list = ['output']
            outputs = {'Output': ['output']}
            outputs_dtype = {'Output': input_output_data_type}
            if not is_input_lod_tensor:
                outputs['OutputLength'] = ['output_length']
                outputs_tensor_name_list.append('output_length')
                outputs_dtype['OutputLength'] = input_output_data_type
            return inputs, attrs, outputs, inputs_tensor, outputs_tensor_name_list, outputs_dtype

        inputs, attrs, outputs, inputs_tensor, outputs_tensor_name_list, outputs_dtype = gen_op_inputs_outputs_attrs(
        )
        ctc_align_op = OpConfig(
            type="ctc_align",
            inputs=inputs,
            outputs=outputs,
            attrs=attrs,
            outputs_dtype=outputs_dtype)

        program_config = ProgramConfig(
            ops=[ctc_align_op],
            weights={},
            inputs=inputs_tensor,
            outputs=outputs_tensor_name_list)
        return program_config

    def sample_predictor_configs(self):
        return self.get_predictor_configs(), ["ctc_align"], (1e-5, 1e-5)

    def add_ignore_pass_case(self):
        pass

    def test(self, *args, **kwargs):
        self.run_and_statis(quant=False, max_examples=300)


if __name__ == "__main__":
    unittest.main(argv=[''])
