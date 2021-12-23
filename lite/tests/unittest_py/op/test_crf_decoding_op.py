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


class TestCrfDecodingOp(AutoScanTest):
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
        # LoD tensor need
        sequence_num = draw(st.integers(min_value=1, max_value=20))
        tag_num = draw(st.integers(min_value=10, max_value=20))
        max_sequence_length = draw(st.integers(min_value=10, max_value=10))

        def gen_lod_data(sequence_num, max_sequence_length):
            lod_data_init_list = [
                np.random.randint(1, max_sequence_length + 1)
                for i in range(sequence_num)
            ]
            lod_data_init_list.insert(0, 0)
            lod_data_init_array = np.array(lod_data_init_list)
            lod_data = np.cumsum(lod_data_init_array)
            return [lod_data.tolist()]

        has_label_input_flag = draw(st.booleans())
        input_lod_tensor_flag = draw(st.booleans())

        if input_lod_tensor_flag:
            emission_lod_info_data = gen_lod_data(sequence_num,
                                                  max_sequence_length)
        else:
            emission_lod_info_data = None

        def gen_input_emission_data():
            if input_lod_tensor_flag:
                total_sequence_length_of_mini_batch = emission_lod_info_data[
                    -1][-1]
                emission_data = np.random.uniform(
                    -1, 1, [total_sequence_length_of_mini_batch,
                            tag_num]).astype(np.float32)
            else:
                emission_data = np.random.uniform(
                    -1, 1, [sequence_num, max_sequence_length,
                            tag_num]).astype(np.float32)
            return emission_data

        def gen_input_transition_data():
            return np.random.uniform(-0.5, 0.5,
                                     [tag_num + 2, tag_num]).astype(np.float32)

        def gen_input_length_data():
            input_length_data = [
                np.random.randint(1, max_sequence_length + 1)
                for i in range(sequence_num)
            ]
            return np.array(input_length_data).reshape(-1, 1).astype(np.int64)

        def gen_input_label_data():
            if input_lod_tensor_flag:
                total_sequence_length_of_mini_batch = emission_lod_info_data[
                    -1][-1]
                label_data = np.random.randint(
                    0, tag_num,
                    [total_sequence_length_of_mini_batch, 1]).astype(np.int64)
            else:
                label_data = np.random.randint(
                    0, tag_num,
                    [sequence_num, max_sequence_length]).astype(np.int64)
            return label_data

        def gen_inputs():
            inputs = {
                "Emission": ["input_emission_data"],
                "Transition": ["input_transition_data"]
            }
            inputs_tensor = {
                "input_emission_data": TensorConfig(
                    data_gen=partial(gen_input_emission_data),
                    lod=emission_lod_info_data),
                "input_transition_data":
                TensorConfig(data_gen=partial(gen_input_transition_data))
            }
            if has_label_input_flag:
                inputs['Label'] = ["input_label_data"]
                inputs_tensor['input_label_data'] = TensorConfig(
                    data_gen=partial(gen_input_label_data),
                    lod=emission_lod_info_data)
            if not input_lod_tensor_flag:
                inputs['Length'] = ['input_length_data']
                inputs_tensor['input_length_data'] = TensorConfig(
                    data_gen=partial(gen_input_length_data))
            return inputs, inputs_tensor

        inputs, inputs_tensor = gen_inputs()
        crf_decoding_op = OpConfig(
            type="crf_decoding",
            inputs=inputs,
            outputs={"ViterbiPath": ["viterbi_path"]},
            attrs={})
        crf_decoding_op.outputs_dtype = {"viterbi_path": np.int64}
        program_config = ProgramConfig(
            ops=[crf_decoding_op],
            weights={},
            inputs=inputs_tensor,
            outputs=["viterbi_path"])
        return program_config

    def sample_predictor_configs(self):
        return self.get_predictor_configs(), ["crf_decoding"], (1e-5, 1e-5)

    def add_ignore_pass_case(self):
        pass

    def test(self, *args, **kwargs):
        self.run_and_statis(quant=False, max_examples=300)


if __name__ == "__main__":
    unittest.main(argv=[''])
