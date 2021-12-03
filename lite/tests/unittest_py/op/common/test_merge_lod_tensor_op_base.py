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
    def generate_input_data_in_shape(*args, **kwargs):
        return np.arange(10).reshape(10, 1).astype('int32')
    def generate_mask_data_in_shape(*args, **kwargs):
        return np.expand_dims(np.array([0, 0, 1, 1, 1, 1, 0, 0, 0, 0]).astype('bool'), axis=1)
    def generate_in_true_data_in_shape(*args, **kwargs):
        return np.expand_dims(np.array([2, 3, 4, 5]).astype('int32'), axis=1)
    def generate_in_false_data_in_shape(*args, **kwargs):
        return np.expand_dims(np.array([0, 1, 6, 7, 8, 9]).astype('int32'), axis=1)
    use_merge_lod_infer = draw(st.sampled_from(["false"]))
    match_matrix_tensor_op = OpConfig(
        type = "merge_lod_tensor",
        inputs = {"X" : ["input_data_x"], "Mask": ["Mask"], "InTrue" : ["InTrue"], "InFalse":["InFalse"]},
        outputs = {"Out": ["output_data"]},
        attrs = {"level":0, "use_merge_lod_infer":["use_merge_lod_infer"]})
    program_config = ProgramConfig(
        ops=[match_matrix_tensor_op],
        weights={},
        inputs={
            "input_data_x":
            TensorConfig(data_gen = partial(generate_input_data_in_shape)),
            "Mask":
            TensorConfig(data_gen = partial(generate_mask_data_in_shape)),
            "InTrue" : TensorConfig(data_gen = generate_in_true_data_in_shape),
            "InFalse" : TensorConfig(data_gen = generate_in_false_data_in_shape)
        },
        outputs={"output_data"})
    return program_config
