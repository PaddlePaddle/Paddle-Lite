# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import os
import numpy as np
import paddle
import paddle.fluid as fluid
from paddle.fluid import core

paddle.enable_static()

SRC_MODEL_DIR = "./simple_model"
DST_MODEL_DIR = "./output_model"
MODEL_FILE = "model.pdmodel"
PARAMS_FILE = "model.pdiparams"

#MODEL_FILE = ""
#PARAMS_FILE = ""


def insert_head_transpose(block, in_var_name, out_var_name, out_var_shape):
    in_var = block.var(in_var_name)
    out_var = block.create_var(
        name=out_var_name,
        shape=out_var_shape,
        dtype=in_var.dtype,
        type=in_var.type,
        persistable=False,
        stop_gradient=True)
    for i in range(len(block.ops)):
        found = False
        op_desc = block.ops[i].desc
        op_type = op_desc.type()
        for arg_name in op_desc.inputs():
            input_names = op_desc.input(arg_name)
            for input_name in input_names:
                if input_name == in_var_name:
                    found = True
                    print("Found %s" % op_type)
                    transpose_op = block._insert_op(
                        i,
                        type="transpose",
                        inputs={"X": in_var},
                        outputs={"Out": out_var},
                        attrs={"axis": [0, 3, 1, 2]})
                    op_desc.set_input("X", [out_var_name])
                    break
            if found:
                break
        if found:
            break


def insert_tail_transpose(block, in_var_name, out_var_name, in_var_shape):
    out_var = block.var(out_var_name)
    in_var = block.create_var(
        name=in_var_name,
        shape=in_var_shape,
        dtype=out_var.dtype,
        type=out_var.type,
        persistable=False,
        stop_gradient=True)
    for i in range(len(block.ops)):
        found = False
        op_desc = block.ops[i].desc
        op_type = op_desc.type()
        for arg_name in op_desc.outputs():
            output_names = op_desc.output(arg_name)
            for output_name in output_names:
                if output_name == out_var_name:
                    found = True
                    print("Found %s" % op_type)
                    transpose_op = block._insert_op(
                        i + 1,
                        type="transpose",
                        inputs={"X": in_var},
                        outputs={"Out": out_var},
                        attrs={"axis": [0, 2, 3, 1]})
                    op_desc.set_output("Out", [in_var_name])
                    break
            if found:
                break
        if found:
            break


def nchw_to_nhwc(nchw_dims):
    p0, p1, p2, p3 = nchw_dims
    return [p0, p2, p3, p1]


def main(argv=None):
    place = paddle.CPUPlace()
    exe = paddle.static.Executor(place=place)
    if len(MODEL_FILE) == 0 and len(PARAMS_FILE) == 0:
        [program, feed_target_names,
         fetch_targets] = fluid.io.load_inference_model(SRC_MODEL_DIR, exe)
    else:
        [program, feed_target_names,
         fetch_targets] = fluid.io.load_inference_model(
             SRC_MODEL_DIR,
             exe,
             model_filename=MODEL_FILE,
             params_filename=PARAMS_FILE)
    print('--- origin feed_target_names ---')
    print(feed_target_names)
    print('--- origin fetch_targets ---')
    print(fetch_targets)
    try:
        os.makedirs(DST_MODEL_DIR)
    except OSError as e:
        if e.errno != 17:
            raise
    # Find an op whose output is the specified variable, then insert a transpose op before/after it.
    in0_name = "image"
    out0_name = "batch_norm_0.tmp_3"
    in0_shape = [1, 3, 224, 224]
    out0_shape = [1, 32, 112, 112]
    new_in0_name = in0_name + "_transpose"
    new_out0_name = out0_name + "_transpose"
    new_in0_shape = nchw_to_nhwc(in0_shape)
    new_out0_shape = nchw_to_nhwc(out0_shape)
    block = program.block(0)
    in0_var = block.var(in0_name)
    in0_var.desc.set_shape(new_in0_shape)
    out0_var = block.var(out0_name)
    out0_var.desc.set_shape(new_out0_shape)
    insert_head_transpose(block, in0_name, new_in0_name, in0_shape)
    insert_tail_transpose(block, new_out0_name, out0_name, out0_shape)
    if len(MODEL_FILE) == 0 and len(PARAMS_FILE) == 0:
        fluid.io.save_inference_model(
            DST_MODEL_DIR,
            feed_target_names,
            fetch_targets,
            exe,
            main_program=program)
    else:
        fluid.io.save_inference_model(
            DST_MODEL_DIR,
            feed_target_names,
            fetch_targets,
            exe,
            main_program=program,
            model_filename=MODEL_FILE,
            params_filename=PARAMS_FILE)
    print("Done.")


if __name__ == '__main__':
    main()
