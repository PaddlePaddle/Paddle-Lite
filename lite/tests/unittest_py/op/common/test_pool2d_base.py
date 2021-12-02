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
    in_shape = draw(st.lists(st.integers(
            min_value=1, max_value=10), min_size=4, max_size=4))
    pool_type = draw(st.sampled_from(["max", "avg"]))
    padding_algorithm = draw(st.sampled_from(["SAME", "VALID"]))
    pool_padding = draw(st.sampled_from([[0, 0], [0, 0, 1, 1], [1,1,1,1],[1,1]]))
    global_pooling = draw(st.sampled_from([True, False]))
    exclusive = draw(st.sampled_from([True, False]))
    ceil_mode = draw(st.sampled_from([True, False]))
    pool_size = draw(st.sampled_from([[1],[2],[3]]))
    pool_stride = draw(st.sampled_from([[1],[2]]))

    if padding_algorithm == "VALID" or padding_algorithm == "SAME":
        pool_padding = [0, 0]


    def generate_input(*args, **kwargs):
        return np.random.random(in_shape).astype(np.float32)
       
    build_ops = OpConfig(
        type = "pool2d",
        inputs={"X": ["input_data"]},
        outputs={"Out": ["output_data"]},
        attrs={
            "pooling_type": pool_type,
            "ksize": pool_size,
            "global_pooling": global_pooling,
            "strides": pool_stride,
            "paddings": pool_padding,
            "padding_algorithm": padding_algorithm,
            "use_cudnn": False,
            "ceil_mode": ceil_mode,
            "use_mkldnn": False,
            "exclusive": exclusive,
            "data_format": "NCHW",
        })
    program_config = ProgramConfig(
        ops=[build_ops],
        weights={},
        inputs={
            "input_data":
            TensorConfig(data_gen=partial(generate_input)),
        },
        outputs=["output_data"])
    return program_config
