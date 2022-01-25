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
from hypothesis import assume
import hypothesis.strategies as st


def sample_program_configs(draw):
    in_shape = draw(
        st.lists(
            st.integers(
                min_value=5, max_value=8), min_size=4, max_size=4))
    starts_data = draw(
        st.lists(
            st.integers(
                min_value=0, max_value=2), min_size=1, max_size=4))
    ends_data = draw(
        st.lists(
            st.integers(
                min_value=3, max_value=4), min_size=1, max_size=4))
    strides_data = draw(
        st.lists(
            st.integers(
                min_value=1, max_value=1), min_size=1, max_size=4))
    axes_data = draw(
        st.lists(
            st.integers(
                min_value=0, max_value=3), min_size=1, max_size=4))
    # whether this axis for runtime calculations
    infer_flags_data = draw(
        st.lists(
            st.integers(
                min_value=1, max_value=1), min_size=1, max_size=4))

    assume(len(starts_data) == len(ends_data))
    assume(len(ends_data) == len(strides_data))
    assume(len(strides_data) == len(axes_data))
    assume(len(axes_data) == len(infer_flags_data))

    def generate_StartsTensorList_data():
        return np.array(starts_data).astype("int32")

    def generate_EndsTensorList_data():
        return np.array(ends_data).astype("int32")

    def generate_StridesTensorList_data():
        return np.array(strides_data).astype("int32")

    def generate_StartsTensor_data():
        return np.array(starts_data).astype("int32")

    def generate_EndsTensor_data():
        return np.array(ends_data).astype("int32")

    def generate_StridesTensor_data():
        return np.array(strides_data).astype("int32")

    strideslice_op = OpConfig(
        type="strided_slice",
        inputs={"Input": ["input_data"]},
        outputs={"Out": ["output_data"]},
        attrs={
            "starts": starts_data,
            "ends": ends_data,
            "strides": strides_data,
            "axes": axes_data,
            "infer_flags": infer_flags_data,
            "decrease_axis": []
        })
    program_config = ProgramConfig(
        ops=[strideslice_op],
        weights={},
        inputs={
            "input_data": TensorConfig(shape=in_shape),
            "StartsTensorList_data":
            TensorConfig(data_gen=partial(generate_EndsTensorList_data)),
            "EndsTensorList_data":
            TensorConfig(data_gen=partial(generate_StartsTensorList_data)),
            "StridesTensorList_data":
            TensorConfig(data_gen=partial(generate_StridesTensorList_data)),
            "StartsTensor_data":
            TensorConfig(data_gen=partial(generate_EndsTensor_data)),
            "EndsTensor_data":
            TensorConfig(data_gen=partial(generate_StartsTensor_data)),
            "StridesTensor_data":
            TensorConfig(data_gen=partial(generate_StridesTensor_data)),
        },
        outputs=["output_data"])
    return program_config
