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
    priorbox_shape = draw(
        st.sampled_from([[30, 4], [80, 4], [70, 4], [66, 4]]))
    code_type = draw(
        st.sampled_from(["encode_center_size", "decode_center_size"]))
    axis = draw(st.sampled_from([0, 1]))
    box_normalized = draw(st.booleans())
    variance = draw(st.sampled_from([[0.1, 0.2, 0.3, 0.4], []]))
    lod_data = [[1, 1, 1, 1, 1]]

    if code_type == "encode_center_size":
        targetbox_shape = draw(st.sampled_from([[30, 4], [80, 4]]))
    else:
        num0 = 1
        num1 = 1
        num2 = 1
        if axis == 0:
            num1 = priorbox_shape[0]
            num0 = np.random.randint(1, 100)
        else:
            num0 = priorbox_shape[0]
            num1 = np.random.randint(1, 100)
        num2 = priorbox_shape[1]
        targetbox_shape = draw(st.sampled_from([[num0, num1, num2]]))

    def generate_priorbox(*args, **kwargs):
        return np.random.random(priorbox_shape).astype(np.float32)

    def generate_priorbox_var(*args, **kwargs):
        return np.random.random(priorbox_shape).astype(np.float32)

    def generate_targetbox(*args, **kwargs):
        return np.random.random(targetbox_shape).astype(np.float32)

    box_coder_op = OpConfig(
        type="box_coder",
        inputs={
            "PriorBox": ["priorbox_data"],
            "TargetBox": ["targetbox_data"],
            "PriorBoxVar": ["priorbox_var_data"]
        },
        outputs={"OutputBox": ["outputbox_data"]},
        attrs={
            "code_type": code_type,
            "box_normalized": box_normalized,
            "axis": axis,
            "variance": variance
        })

    program_config = ProgramConfig(
        ops=[box_coder_op],
        weights={},
        inputs={
            "priorbox_data": TensorConfig(
                data_gen=partial(generate_priorbox), lod=lod_data),
            "priorbox_var_data":
            TensorConfig(data_gen=partial(generate_priorbox_var)),
            "targetbox_data": TensorConfig(
                data_gen=partial(generate_targetbox), lod=lod_data),
        },
        outputs=["outputbox_data"])
    return program_config
