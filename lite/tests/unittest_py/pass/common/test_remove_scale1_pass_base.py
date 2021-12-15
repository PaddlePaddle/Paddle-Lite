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
from hypothesis import given, settings, seed, example, assume, reproduce_failure
import hypothesis.strategies as st


def sample_program_configs(draw):
    in_shape = draw(
        st.lists(
            st.integers(
                min_value=1, max_value=8), min_size=1, max_size=4))
    abs_op = OpConfig(
        type="assign",
        inputs={"X": ["input_data"]},
        outputs={"Out": ["abs_output_data"]},
        attrs={})

    scale_op = OpConfig(
        type="scale",
        inputs={"X": ["abs_output_data"]},
        outputs={"Out": ["output_data"]},
        attrs={"scale": 1.0,
               "bias": 0.0,
               "bias_after_scale": True})

    ops = [abs_op, scale_op]
    program_config = ProgramConfig(
        ops=ops,
        weights={},
        inputs={"input_data": TensorConfig(shape=in_shape)},
        outputs=["output_data"])
    return program_config
