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
from hypothesis import assume
import hypothesis.strategies as st

def sample_program_configs(draw):
    in_shape = draw(st.lists(st.integers(min_value=1, max_value=8), min_size = 4, max_size=4))
    is_test_data = draw(st.sampled_from([True]))
    use_global_stats_data = draw(st.sampled_from([False]))
    epsilon_data = draw(st.floats(min_value=0.00001, max_value=0.001))
    momentum_data = draw(st.floats(min_value=0.1, max_value=0.9))
    data_layout_data = draw(st.sampled_from(["NCHW"]))

    # sync_batch_norm is not supported on cpu by paddle
    # so we test batch_norm
    sync_batch_norm_op = OpConfig(
        type = "batch_norm",
        inputs = {"X" : ["X_data"],
                  "Bias" : ["Bias_data"],
                  "Mean" : ["Mean_data"],
                  "Scale" : ["Scale_data"],
                  "Variance" : ["Variance_data"]
                  },
        outputs = {"Y": ["Y_data"],
                  "MeanOut": ["Mean_data"],
                  #"ReserveSpace": ["ReserveSpace_data"],
                  "VarianceOut": ["Variance_data"],
                  "SavedMean": ["SavedMean_data"],
                  "SavedVariance": ["SavedVariance_data"]},
        attrs = {"is_test":is_test_data,
                 "use_global_stats":use_global_stats_data,
                 "epsilon":epsilon_data,
                 "momentum":momentum_data,
                 "data_layout": data_layout_data,
                 "trainable_statistics": False
                })
    program_config = ProgramConfig(
        ops=[sync_batch_norm_op],
        weights={
            "Mean_data": TensorConfig(shape=in_shape[1:2]),
            "Variance_data" : TensorConfig(shape=in_shape[1:2]),
            "SavedMean_data" : TensorConfig(shape=in_shape[1:2]),
            "SavedVariance_data" : TensorConfig(shape=in_shape[1:2]),
        },
        inputs={
            "X_data": TensorConfig(shape=in_shape),
            "Bias_data": TensorConfig(shape=in_shape[1:2]),
            "Scale_data": TensorConfig(shape=in_shape[1:2]),
        },
        outputs=["Y_data", "Mean_data", "Variance_data",
                "SavedMean_data", "SavedVariance_data"])
    return program_config
