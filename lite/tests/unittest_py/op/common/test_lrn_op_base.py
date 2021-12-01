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
    in_shape = draw(st.lists(st.integers(min_value=1, max_value=8), min_size = 4, max_size=4))
    n_ = draw(st.sampled_from([5,7, 9]))
    k_ = draw(st.sampled_from([0.9, 1., 1.1]))
    alpha_ = draw(st.sampled_from([0.9, 1., 1.1]))
    beta_ = draw(st.sampled_from([0.5, 0.75, 1.]))
    data_format_ = draw(st.sampled_from(["NCHW"]))
    lrn_op = OpConfig(
        type = "lrn",
       inputs = {"X" : ["input_data"]},
        outputs = {"Out": ["output_data"], "MidOut": ["output_data_mid"]},
        attrs = {"n":n_, "k":k_, "alpha":alpha_, "beta":beta_, "data_format":"NCHW", "is_test":1})
    program_config = ProgramConfig(
        ops=[lrn_op],
        weights={},
        inputs={
            "input_data":
            TensorConfig(shape=in_shape)
        },
        outputs=["output_data"])
    return program_config
