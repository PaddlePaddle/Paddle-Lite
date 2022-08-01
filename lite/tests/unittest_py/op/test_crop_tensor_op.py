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


class TestCropTensorOp(AutoScanTest):
    def __init__(self, *args, **kwargs):
        AutoScanTest.__init__(self, *args, **kwargs)
        self.enable_testing_on_place(
            TargetType.Host,
            PrecisionType.FP32,
            DataLayoutType.Any,
            thread=[1, 4])

    def is_program_valid(self,
                         program_config: ProgramConfig,
                         predictor_config: CxxConfig) -> bool:
        return True

    def sample_program_configs(self, draw):
        '''
            Note: Current not support ShapeTensor and OffsetsTensor
        '''
        shape_value_min = 1
        shape_value_max = 25
        input_data_x_shape = draw(
            st.lists(
                st.integers(
                    min_value=shape_value_min, max_value=shape_value_max),
                min_size=1,
                max_size=6))
        input_data_x_dtype = draw(st.sampled_from([np.float32, np.int32]))
        x_dims_size = len(input_data_x_shape)
        offsets = [
            np.random.randint(
                low=0, high=input_data_x_shape[i]) for i in range(x_dims_size)
        ]
        shape = [
            input_data_x_shape[i] - offsets[i] for i in range(x_dims_size)
        ]

        def gen_input_x_data(*args, **kwargs):
            return np.random.randint(0, 20,
                                     kwargs['shape']).astype(kwargs['dtype'])

        def gen_input_offsets_data():
            return np.array(offsets).astype(np.int32)

        def gen_input_shape_data():
            return np.array(shape).astype(np.int32)

        def GenOpInputsAndAttrs():
            def GenOpInputs():
                inputs = {"X": ["input_data_x"]}
                inputs_tensor = {
                    "input_data_x": TensorConfig(data_gen=partial(
                        gen_input_x_data,
                        shape=input_data_x_shape,
                        dtype=input_data_x_dtype))
                }
                if draw(st.booleans()):
                    inputs["Shape"] = ["input_data_shape"]
                    inputs_tensor["input_data_shape"] = TensorConfig(
                        data_gen=partial(gen_input_shape_data))
                if draw(st.booleans()):
                    inputs["Offsets"] = ["input_data_offsets"]
                    inputs_tensor["input_data_offsets"] = TensorConfig(
                        data_gen=partial(gen_input_offsets_data))
                return inputs, inputs_tensor

            inputs, inputs_tensor = GenOpInputs()

            def GenOpAttrs():
                attrs = {"offsets": offsets, "shape": shape}
                if "Offsets" in inputs:
                    attrs['offsets'] = []
                return attrs

            attrs = GenOpAttrs()
            return inputs, inputs_tensor, attrs

        inputs, inputs_tensor, attrs = GenOpInputsAndAttrs()
        crop_tensor_op = OpConfig(
            type="crop_tensor",
            inputs=inputs,
            outputs={"Out": ["output_data"]},
            attrs=attrs)
        program_config = ProgramConfig(
            ops=[crop_tensor_op],
            weights={},
            inputs=inputs_tensor,
            outputs=["output_data"])
        return program_config

    def sample_predictor_configs(self):
        config = CxxConfig()
        return self.get_predictor_configs(), ["crop_tensor"], (1e-5, 1e-5)

    def add_ignore_pass_case(self):
        pass

    def test(self, *args, **kwargs):
        self.run_and_statis(quant=False, max_examples=300)


if __name__ == "__main__":
    unittest.main(argv=[''])
