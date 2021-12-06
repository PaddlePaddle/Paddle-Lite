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
import hypothesis.strategies as st

def sample_program_configs(draw):
    #ix = st.integers(min_value=1, max_value=32)
    #iy = st.integers(min_value=1, max_value=32)
    #h = st.integers(min_value=1, max_value=32)
    #dim_t = st.integers(min_value=1, max_value=32)
    ix, iy, h, dim_t = [5, 8, 20, 4]
    def generate_x_data_in_shape(*args, **kwargs):
        return np.random.random((ix, h)).astype('float32')
    def generate_y_data_in_shape(*args, **kwargs):
        return np.random.random((iy, h)).astype('float32')
    def generate_w_data_in_shape(*args, **kwargs):
        return np.random.random((h, dim_t, h)).astype('float32')
    in_shape_x = draw(st.sampled_from([[ix, h]]))
    in_shape_y = draw(st.sampled_from([[iy, h]]))
    input_lod_x = draw(st.sampled_from([[[1, 2, 2]]]))
    input_lod_y = draw(st.sampled_from([[[3, 1, 4]]]))
    input_weight_shape = draw(st.sampled_from([[h, dim_t, h]]))
    match_matrix_tensor_op = OpConfig(
        type = "match_matrix_tensor",
        inputs = {"X" : ["input_data_x"], "Y" : ["input_data_y"], "W" : ["W"]},
        outputs = {"Out" : ["output_data"], "tmp" : ["tmp_data"]},
        attrs = {"dim_t" : dim_t})
    program_config = ProgramConfig(
        ops=[match_matrix_tensor_op],
        weights={},
        inputs={
            "input_data_x":
            TensorConfig(data_gen=partial(generate_x_data_in_shape), lod = input_lod_x),
            "input_data_y":
            TensorConfig(data_gen=partial(generate_y_data_in_shape), lod = input_lod_y),
            "W" : TensorConfig(data_gen=partial(generate_w_data_in_shape))
        },
        outputs={"output_data", "tmp_data"})
    return program_config
