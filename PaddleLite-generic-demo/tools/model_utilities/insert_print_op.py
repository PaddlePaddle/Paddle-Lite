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
    exe.run(paddle.static.default_startup_program())
    print('--- feed_target_names ---')
    print(feed_target_names)
    print('--- fetch_targets ---')
    print(fetch_targets)
    try:
        os.makedirs(DST_MODEL_DIR)
    except OSError as e:
        if e.errno != 17:
            raise
    # Specify a named variable
    output_name = "conv2d_0.tmp_0"
    # Find an op whose output is the specified variable, then insert a print op after it.
    input_name = "print_0.tmp_0"
    main_block = program.block(0)
    output = main_block.var(output_name)
    input = main_block.create_var(
        name=input_name,
        shape=output.shape,
        dtype=output.dtype,
        type=output.type,
        persistable=False,
        stop_gradient=False)
    for i in range(len(main_block.ops)):
        op_desc = main_block.ops[i].desc
        op_type = op_desc.type()
        if op_type == "print":
            continue
        found = False
        for arg_name in op_desc.outputs():
            out_names = op_desc.output(arg_name)
            for j in range(len(out_names)):
                if out_names[j] == output_name:
                    print("Found op %s!" % op_type)
                    found = True
                    out_names[j] = input_name
                    op_desc.set_output(arg_name, out_names)
                    # append_op should not be used as it may be incorrectly pruned.
                    main_block._insert_op(
                        index=i + 1,
                        type='print',
                        inputs={'In': input},
                        outputs={'Out': output},
                        attrs={
                            'first_n':
                            -1,  # Only print `first_n` number of times, -1 means print at each time.
                            'summarize':
                            50,  # The number of element to print, -1 means print all elements.
                            'message': output_name,
                            'print_tensor_name': True,
                            'print_tensor_type': True,
                            'print_tensor_shape': True,
                            'print_tensor_layout': True,
                            'print_tensor_lod': True,
                            'print_phase': "FORWARD",
                            'is_forward': True,
                            'op_role': 0
                        })
                    break
            if found:
                break
        if found:
            break
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
