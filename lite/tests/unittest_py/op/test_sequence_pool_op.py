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
import random


class TestSequenceReshapeOp(AutoScanTest):
    def __init__(self, *args, **kwargs):
        AutoScanTest.__init__(self, *args, **kwargs)
        self.enable_testing_on_place(
            TargetType.X86, [PrecisionType.FP32],
            DataLayoutType.NCHW,
            thread=[1, 4])
        self.enable_testing_on_place(
            TargetType.ARM, [PrecisionType.FP32],
            DataLayoutType.NCHW,
            thread=[1, 4])
        self.enable_testing_on_place(
            TargetType.ARM, [PrecisionType.FP16],
            DataLayoutType.NCHW,
            thread=[1, 4])

    def is_program_valid(self,
                         program_config: ProgramConfig,
                         predictor_config: CxxConfig) -> bool:
        return True

    def sample_program_configs(self, draw):
        def generate_lod(seq_num):
            sum = 0
            lod_tensor = []
            lod_tensor.append(sum)
            for i in range(0, seq_num):
                sum += random.randint(1, 9) % 2 + 1
                lod_tensor.append(int(sum))
            return [lod_tensor]

        in_shape = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=4), min_size=2, max_size=8))
        pad_value = draw(st.sampled_from([0.0, 0.2, 0.5, 1.0]))
        pooltype = draw(
            st.sampled_from(
                ["AVERAGE", "SUM", "SQRT", "LAST", "FIRST", "MAX"]))
        seq_num = draw(st.sampled_from([1, 3, 5]))

        lod_tensor = generate_lod(seq_num)
        in_shape0 = draw(
            st.integers(
                min_value=lod_tensor[0][len(lod_tensor[0]) - 1],
                max_value=lod_tensor[0][len(lod_tensor[0]) - 1] + 8))
        in_shape.insert(0, in_shape0)

        lod_level = draw(st.sampled_from([1, 2]))

        if lod_level == 2:
            in_shape = draw(
                st.lists(
                    st.integers(
                        min_value=1, max_value=6),
                    min_size=2,
                    max_size=4))
            in_shape = [4] + in_shape
            lod_tensor = draw(
                st.sampled_from([[[0, 2, 3], [0, 1, 2, 4]],
                                 [[0, 3, 4], [0, 2, 2, 3, 4]]]))

        def generate_input(*args, **kwargs):
            return np.random.random(in_shape).astype(np.float32)

        ops_config = OpConfig(
            type="sequence_pool",
            inputs={"X": ["input_data"], },
            outputs={"Out": ["output_data"],
                     "MaxIndex": ["maxindex"]},
            attrs={"pad_value": pad_value,
                   "pooltype": pooltype})

        outputs_ = ["output_data"]
        if pooltype == "MAX":
            outputs_ = ["output_data", "maxindex"]

        program_config = ProgramConfig(
            ops=[ops_config],
            weights={},
            inputs={
                "input_data": TensorConfig(
                    data_gen=partial(generate_input), lod=lod_tensor)
            },
            outputs=outputs_)

        return program_config

    def sample_predictor_configs(self):
        atol, rtol = 1e-5, 1e-5
        config_lists = self.get_predictor_configs()
        for config in config_lists:
            if config.precision() in [PrecisionType.FP16]:
                atol, rtol = 1e-3, 1e-3

        return self.get_predictor_configs(), ["sequence_pool"], (atol, rtol)

    def add_ignore_pass_case(self):
        def _teller1(program_config, predictor_config):
            target_type = predictor_config.target()
            pooltype = program_config.ops[0].attrs["pooltype"]
            if target_type == TargetType.ARM and predictor_config.precision(
            ) == PrecisionType.FP16 and pooltype == "MAX":
                return True

        self.add_ignore_check_case(
            _teller1, IgnoreReasons.ACCURACY_ERROR,
            "Lite has diff in a specific case in the 2th output on arm for MAX pooling type. but We needn't fix it."
        )

    def test(self, *args, **kwargs):
        self.run_and_statis(quant=False, max_examples=500)


if __name__ == "__main__":
    unittest.main(argv=[''])
