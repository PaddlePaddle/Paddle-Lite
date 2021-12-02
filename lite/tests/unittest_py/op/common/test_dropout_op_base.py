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
    input_data_x_shape = draw(st.lists(st.integers(min_value=1, max_value=8), min_size = 4, max_size = 4))
    dropout_prob = draw(st.floats(min_value=0, max_value=1))
    seed = draw(st.integers())
    dropout_implementation = draw(st.sampled_from(['downgrade_in_infer', 'upscale_in_train']))
    is_test = draw(st.booleans())
    fix_seed = draw(st.booleans())

    def gen_input_data_seed():
        return np.array([seed]).astype(np.int32)

    def GenOpInputs():
        '''
            AsDispensable { "X": False, "Seed": True }
        '''
        inputs = {"X" : ["input_data_x"]}
        inputs_tensor = {"input_data_x" : TensorConfig(shape=input_data_x_shape)}
        if draw(st.booleans()):
            inputs["Seed"] = ["input_data_seed"]
            inputs_tensor["input_data_seed"] = TensorConfig(data_gen=partial(gen_input_data_seed))
        return inputs, inputs_tensor
    
    inputs, inputs_tensor = GenOpInputs()
    dropout_op = OpConfig(
        type = "dropout",
        inputs = inputs,
        outputs = {"Out": ["output_data"],
                   "Mask": ["mask_data"]},
        attrs = {"dropout_prob" : dropout_prob,
                 "fix_seed" : fix_seed,
                 "seed" : seed,
                 "dropout_implementation" : dropout_implementation,
                 "is_test" : is_test})
    program_config = ProgramConfig(
        ops=[dropout_op],
        weights={},
        inputs=inputs_tensor,
        outputs=["output_data", "mask_data"])
    return program_config
