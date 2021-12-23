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
    in_shape = draw(
        st.sampled_from([[3, 6, 4], [5, 7, 4], [4, 8, 4], [10, 10, 4]]))
    iminfo_shape = draw(st.sampled_from([[3, 3], [6, 3], [8, 3]]))
    lod_data = [[1, 2, 3]]

    def generate_input(*args, **kwargs):
        return np.random.random(in_shape).astype(np.float32)

    def generate_iminfo(*args, **kwargs):
        return np.random.random(iminfo_shape).astype(np.float32)

    box_clip_op = OpConfig(
        type="box_clip",
        inputs={"Input": ["input_data"],
                "ImInfo": ["iminfo_data"]},
        outputs={"Output": ["output_data"]},
        attrs={})

    program_config = ProgramConfig(
        ops=[box_clip_op],
        weights={},
        inputs={
            "input_data": TensorConfig(
                data_gen=partial(generate_input), lod=lod_data),
            "iminfo_data": TensorConfig(
                data_gen=partial(generate_iminfo), lod=lod_data),
        },
        outputs=["output_data"])
    return program_config
