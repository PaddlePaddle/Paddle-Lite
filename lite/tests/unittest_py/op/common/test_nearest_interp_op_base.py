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
    X_shape = draw(st.sampled_from([[1, 3, 24, 24]]))
    Scale_shape = draw(st.sampled_from([[1]]))
    Tensor_shape = draw(st.sampled_from([[32, 64]]))
    align_corners = draw(st.booleans())
    scale = draw(st.sampled_from([0.5, 0.3, 1.2]))
    interp_method = draw(st.sampled_from(["nearest"]))
    out_w = draw(st.sampled_from([32, 64]))
    out_h = draw(st.sampled_from([32, 64]))
    data_layout = draw(st.sampled_from(["NCHW"]))

    nearest_interp = OpConfig(
        type = "nearest_interp",
        inputs = {"X" : ["input_data_x"], "Scale":["Scale"]},
        outputs = {"Out": ["output_data"]},
        attrs = {"data_layout":data_layout, "scale":scale, "out_w":out_w,"out_h":out_h, "interp_method":interp_method , "align_corners":align_corners})
    program_config = ProgramConfig(
        ops=[nearest_interp],
        weights={},
        inputs={
            "input_data_x":
            TensorConfig(shape=X_shape),
            "SizeTensor" : TensorConfig(shape=Tensor_shape),
            "Scale" : TensorConfig(shape=Scale_shape)
        },
        outputs={"output_data"})
    return program_config
