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

from program_config import TensorConfig, BlockConfig, ProgramConfig, OpConfig, CxxConfig, TargetType, PrecisionType, DataLayoutType, Place, VarType
import numpy as np
from functools import partial
from typing import Optional, List, Callable, Dict, Any, Set
import unittest
import hypothesis
from hypothesis import assume
import hypothesis.strategies as st


'''
import paddle
paddle.enable_static()

import paddle.fluid as fluid
import numpy as np

main_program = fluid.default_main_program()
startup_program = fluid.default_startup_program()

start = fluid.layers.fill_constant(shape=[1], dtype='int64', value=0)
end = fluid.data(name='loop_end', shape=[1], dtype='int64')
tmp = fluid.layers.fill_constant(shape=[3,2], dtype='int64', value=1)
arr = fluid.layers.array_write(tmp, i=start)

def cond(i, end, arr, tmp):
  return i < end

def body(i, end, arr, tmp):
  i = i + 1
  arr = fluid.layers.array_write(tmp, i)
  return i, end, arr, tmp

start, end, arr, tmp = fluid.layers.while_loop(cond, body, [start, end, arr, tmp])

output, output_index = fluid.layers.tensor_array_to_tensor(input=arr)
exe = fluid.Executor(fluid.CPUPlace())
res = exe.run(main_program, feed={'loop_end':np.array([3]).astype(np.int64)}, fetch_list=[output, output_index])

#print(main_program.to_string(True))
print(res)

fluid.io.save_inference_model(dirname='array_model', feeded_var_names=['loop_end'], target_vars=[output, output_index], executor=exe, model_filename='array.pdmodel', params_filename='array.pdiparams')
'''
 
def sample_program_configs(draw):
    in_shape = draw(st.lists(st.integers(min_value=2, max_value=2), min_size = 4, max_size=4))
    tensor_type = draw(st.sampled_from([np.float32]))
    dtype2int = {np.bool:0, np.int16:1, np.int32:2, np.int64:3, np.float16:4, np.float32:5, np.float64:6, np.uint8:20, np.int8:21}

    tmp_val_assign_op = OpConfig(
        type = "assign",
        inputs={"X":["fill_constant_3.tmp_0"]},
        outputs={"Out":["fill_constant_3.tmp_0"]},
        outputs_dtype = {'fill_constant_3.tmp_0': np.int64},
        attrs = {}
    )
    loop_end_assign_op = OpConfig(
        type = "assign",
        inputs={"X":["loop_end"]},
        outputs={"Out": ["loop_end"]},
        outputs_dtype = {'loop_end': np.int64},
        attrs = {}
    )
    increment_op = OpConfig(
        type = "scale",
        inputs={"X":["fill_constant_1.tmp_0"]},
        outputs={"Out":["tmp_1"]},
        outputs_dtype = {'tmp_1': np.int64},
        attrs={
            'scale':1,
            'bias':1,
            'bias_after_scale':True
        }
    )
    increment_assign_op = OpConfig(
        type = 'assign',
        inputs={'X':['tmp_1']},
        outputs={"Out":['fill_constant_1.tmp_0']},
        outputs_dtype = {'fill_constant_1.tmp_0': np.int64},
        attrs = {}
    )
    less_than_op_in_block = OpConfig(
        type = "less_than",
        inputs = {"X": ["tmp_1"],
                  "Y": ["loop_end"]},
        outputs = {"Out": ["tmp_2"]},
        outputs_dtype = {'tmp_2': np.bool},
        attrs = {'axis' : -1}
    )
    less_than_assign_op = OpConfig(
        type = 'assign',
        inputs={'X':['tmp_2']},
        outputs={"Out":['tmp_0']},
        outputs_dtype = {'tmp_0': np.bool},
        attrs = {}
    )
    write_to_array_op_in_block = OpConfig(
        type = "write_to_array",
        inputs = {"X" : ["fill_constant_3.tmp_0"],
                  "I" : ["tmp_1"]},
        outputs = {"Out" : ["array_write_1.out"]},
        outputs_var_type = {"array_write_1.out": VarType.LOD_TENSOR_ARRAY},
        outputs_dtype = {'array_write_1.out': tensor_type},
        attrs = {}
    )
    write_to_array_assign_op = OpConfig(
        type = 'assign',
        inputs={'X':['array_write_1.out']},
        outputs={"Out":['array_write_0.out']},
        outputs_var_type = {"array_write_0.out": VarType.LOD_TENSOR_ARRAY},
        outputs_dtype = {'array_write_0.out': tensor_type},
        attrs = {}
    )
    sub_block_ops = [increment_op, tmp_val_assign_op, loop_end_assign_op, increment_assign_op, less_than_op_in_block, less_than_assign_op, write_to_array_op_in_block, write_to_array_assign_op]
    sub_block = BlockConfig(
        ops = sub_block_ops,
        vars = ['tmp_1', 'tmp_2', 'array_write_1.out'],
        vars_dtype = {'tmp_1': np.int64, 'tmp_2': np.bool, 'array_write_1.out': tensor_type},
        vars_var_type = {'array_write_1.out': VarType.LOD_TENSOR_ARRAY}
    )
    
    idx_start_op = OpConfig(
        type = "fill_constant",
        inputs = {},
        outputs={'Out': ['fill_constant_1.tmp_0']},
        outputs_dtype = {'fill_constant_1.tmp_0':np.int64},
        attrs={
            'shape': [1],
            'dtype': 3,
            'value': 0,
        })
    fill_constant_op = OpConfig(
        type = "fill_constant",
        inputs = {},
        outputs = {'Out':['fill_constant_3.tmp_0']},
        outputs_dtype = {'fill_constant_3.tmp_0':tensor_type},
        attrs={
            'shape': [3,2],
            'dtype': dtype2int[tensor_type],
            'value': 1
        }
    )
    write_to_array_op = OpConfig(
        type = "write_to_array",
        inputs = {"X" : ["fill_constant_3.tmp_0"],
                  "I" : ["fill_constant_1.tmp_0"]},
        outputs = {"Out" : ["array_write_0.out"]},
        outputs_var_type = {"array_write_0.out": VarType.LOD_TENSOR_ARRAY},
        outputs_dtype = {'array_write_0.out': tensor_type},
        attrs = {}
    )
    less_than_op = OpConfig(
        type = "less_than",
        inputs = {"X": ["fill_constant_1.tmp_0"],
                  "Y": ["loop_end"]},
        outputs = {"Out": ["tmp_0"]},
        outputs_dtype = {'tmp_0': np.bool},
        attrs = {'axis' : -1}
    )
    while_op = OpConfig(
        type = 'while',
        inputs = {"X": ["fill_constant_3.tmp_0", "loop_end", "fill_constant_1.tmp_0"],
                  "Condition": ["tmp_0"]},
        outputs = {"Out": ["tmp_0", "loop_end", "fill_constant_3.tmp_0", "fill_constant_1.tmp_0", "array_write_0.out"],
                   "StepScopes": ['_generated_var_0']},
        outputs_var_type = {'_generated_var_0': VarType.STEP_SCOPES,
                            'array_write_0.out': VarType.LOD_TENSOR_ARRAY},
        outputs_dtype = {
            'tmp_0': np.bool,
            'array_write_0.out': tensor_type,
            'fill_constant_3.tmp_0': tensor_type,
            'loop_end': np.int64,
            'fill_constant_1.tmp_0': np.int64
        },
        attrs = {
            'sub_block' : sub_block,
            'is_test' : True
        }
    )

    tensor_array_to_tensor_op = OpConfig(
        type = "tensor_array_to_tensor",
        inputs = {"X" : ["array_write_0.out"]},
        outputs = {"Out": ["tensor_array_to_tensor_0.tmp_0"],
                  "OutIndex" : ["tensor_array_to_tensor_0.tmp_1"]},
        outputs_dtype = {"tensor_array_to_tensor_0.tmp_1": np.int32,
                         'tensor_array_to_tensor_0.tmp_0': tensor_type},
        attrs = {"axis" : 1,
                 "use_stack" : False,
                })
    ops = [idx_start_op, fill_constant_op, write_to_array_op, less_than_op, while_op, tensor_array_to_tensor_op]
    def generate_loop_end():
        return np.array([3]).astype(np.int64)
    program_config = ProgramConfig(
        ops=ops,
        weights={},
        inputs={
            "loop_end": TensorConfig(data_gen=generate_loop_end)
        },
        outputs=["tensor_array_to_tensor_0.tmp_0","tensor_array_to_tensor_0.tmp_1"])
    return program_config
