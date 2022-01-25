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
    topks = [1, 3]
    channel_num = draw(st.sampled_from([1, 3, 5]))
    dim = draw(st.sampled_from([10, 12]))
    row = draw(st.sampled_from([[30], [40], [50]]))
    col = draw(st.sampled_from([[25], [35], [45]]))
    feature = [row[i] * col[i] for i in range(len(row))]
    lod_ = [[x * channel_num for x in feature]]

    def generate_input(*args, **kwargs):
        return np.arange(sum(feature) * channel_num).astype('float32')

    def generate_row(*args, **kwargs):
        return np.random.random((sum(row), dim)).astype('float32')

    def generate_column(*args, **kwargs):
        return np.random.random((sum(col), dim)).astype('float32')

    ops_config = OpConfig(
        type="sequence_topk_avg_pooling",
        inputs={"X": ["input_data"],
                "ROW": ["row"],
                "COLUMN": ["column"]},
        outputs={"Out": ["output_data"],
                 "pos": ["pos"]},
        attrs={"topks": topks,
               "channel_num": channel_num})

    program_config = ProgramConfig(
        ops=[ops_config],
        weights={},
        inputs={
            "input_data": TensorConfig(
                data_gen=partial(generate_input), lod=lod_),
            "row": TensorConfig(
                data_gen=partial(generate_row), lod=[row]),
            "column": TensorConfig(
                data_gen=partial(generate_column), lod=[col])
        },
        outputs=["output_data", "pos"])

    return program_config
