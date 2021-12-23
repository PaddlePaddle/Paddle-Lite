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
from hypothesis import assume


def sample_program_configs(draw):
    in_shape = draw(st.sampled_from([[1, 4], [4, 1]]))
    ids_shape = draw(st.sampled_from([[4, 3], [4, 2]]))
    is_accumulated = draw(st.sampled_from([True, False]))
    level = draw(st.integers(min_value=1, max_value=10))
    beam_size = draw(st.integers(min_value=1, max_value=10))
    end_id = draw(st.integers(min_value=1, max_value=10))
    lod_data = [[0, 2, 4], [0, 1, 2, 3, 4]]

    def generate_pre_ids(*args, **kwargs):
        return np.random.random(in_shape).astype(np.int64)

    def generate_pre_score(*args, **kwargs):
        return np.random.random(in_shape).astype(np.float32)

    def generate_ids(*args, **kwargs):
        return np.random.random(ids_shape).astype(np.int64)

    def generate_scores(*args, **kwargs):
        return np.random.random(ids_shape).astype(np.float32)

    beam_search_ops = OpConfig(
        type="beam_search",
        inputs={
            "pre_ids": ["pre_ids_data"],
            "pre_scores": ["pre_scores_data"],
            "ids": ["ids_data"],
            "scores": ["scores_data"]
        },
        outputs={
            "selected_ids": ["selected_ids_data"],
            "selected_scores": ["selected_scores_data"],
            "parent_idx": ["parent_idx_data"]
        },
        attrs={
            "level": level,
            "beam_size": beam_size,
            "end_id": end_id,
            "is_accumulated": is_accumulated
        })
    program_config = ProgramConfig(
        ops=[beam_search_ops],
        weights={},
        inputs={
            "pre_ids_data": TensorConfig(data_gen=partial(generate_pre_ids)),
            "pre_scores_data":
            TensorConfig(data_gen=partial(generate_pre_score)),
            "ids_data": TensorConfig(
                data_gen=partial(generate_ids), lod=lod_data),
            "scores_data": TensorConfig(
                data_gen=partial(generate_pre_score), lod=lod_data),
        },
        outputs=[
            "selected_ids_data", "selected_scores_data", "parent_idx_data"
        ])
    return program_config
