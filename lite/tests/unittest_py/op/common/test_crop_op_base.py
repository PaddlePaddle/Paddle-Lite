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
sys.path.append('..')

from program_config import TensorConfig, ProgramConfig, OpConfig, CxxConfig, TargetType, PrecisionType, DataLayoutType, Place
import numpy as np
from functools import partial
from typing import Optional, List, Callable, Dict, Any, Set
import unittest
import hypothesis
from hypothesis import given, settings, seed, example, assume
import hypothesis.strategies as st


def sample_program_configs(draw):
    shape_value_min = 1
    shape_value_max = 25
    input_data_x_shape = draw(
        st.lists(
            st.integers(
                min_value=shape_value_min, max_value=shape_value_max),
            min_size=1,
            max_size=6))
    input_data_y_shape = input_data_x_shape
    x_dims_size = len(input_data_x_shape)

    def gen_input_data_offsets():
        offsets = [
            np.random.randint(
                low=0, high=input_data_x_shape[i]) for i in range(x_dims_size)
        ]
        offsets = np.array(offsets).astype(np.int32)
        return offsets

    def GenOpInputsAndAttrs():
        def GenOpInputs():
            inputs = {"X": ["input_data_x"]}
            inputs_tensor = {
                "input_data_x": TensorConfig(shape=input_data_x_shape)
            }
            if draw(st.booleans()):
                inputs["Y"] = ["input_data_y"]
                inputs_tensor["input_data_y"] = TensorConfig(
                    shape=input_data_y_shape)
            if draw(st.booleans()):
                inputs["Offsets"] = ["input_data_offsets"]
                inputs_tensor["input_data_offsets"] = TensorConfig(
                    data_gen=partial(gen_input_data_offsets))
            return inputs, inputs_tensor

        inputs, inputs_tensor = GenOpInputs()

        def GenOpAttrs():
            offsets = [
                np.random.randint(
                    low=0, high=input_data_x_shape[i])
                for i in range(x_dims_size)
            ]
            shape = [
                input_data_x_shape[i] - offsets[i] for i in range(x_dims_size)
            ]
            attrs = {"offsets": offsets, "shape": shape}
            if "Offsets" in inputs:
                attrs['offsets'] = []
            return attrs

        attrs = GenOpAttrs()
        return inputs, inputs_tensor, attrs

    inputs, inputs_tensor, attrs = GenOpInputsAndAttrs()
    crop_op = OpConfig(
        type="crop",
        inputs=inputs,
        outputs={"Out": ["output_data"]},
        attrs=attrs)

    program_config = ProgramConfig(
        ops=[crop_op],
        weights={},
        inputs=inputs_tensor,
        outputs=["output_data"])
    return program_config
