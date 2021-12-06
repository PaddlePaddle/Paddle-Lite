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
    shape0 = draw(st.integers(min_value=12, max_value=64))
    shape1 = draw(st.sampled_from([4, 8]))
    shape2 = draw(st.integers(min_value=12, max_value=64))
    shape3 = draw(st.integers(min_value=12, max_value=64))
    X_shape = [shape0, shape2, shape1]
    Y_shape = [shape1, shape3, shape2]
    background_label = draw(st.sampled_from([1, 0]))
    score_threshold = draw(st.floats(min_value=1.0, max_value=10.0))
    nms_threshold = draw(st.floats(min_value=1.0, max_value=10.0))
    nms_eta = draw(st.floats(min_value=1.0, max_value=10.0))
    nms_top_k = draw(st.integers(min_value=1, max_value=64))
    keep_top_k = draw(st.integers(min_value=1, max_value=64))
    normalized = draw(st.booleans())

    multiclass_nms_op = OpConfig(
        type = "multiclass_nms",
        inputs = {"BBoxes" : ["input_data_BBoxes"], "Scores" : ["input_data_Scores"]},
        outputs = {"Out": ["output_data"]},
        attrs = {"background_label":background_label, "score_threshold":score_threshold, "nms_threshold":nms_threshold,
           "nms_top_k":nms_top_k, "keep_top_k":keep_top_k, "normalized":normalized, "nms_eta":nms_eta})

    program_config = ProgramConfig(
        ops=[multiclass_nms_op],
        weights={},
        inputs={
            "input_data_BBoxes":
            TensorConfig(shape=X_shape),
            "input_data_Scores" : TensorConfig(shape=Y_shape)
        },
        outputs={"output_data"})
    return program_config
