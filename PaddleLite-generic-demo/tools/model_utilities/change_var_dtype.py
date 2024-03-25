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
    print('--- origin feed_target_names ---')
    print(feed_target_names)
    print('--- origin fetch_targets ---')
    print(fetch_targets)
    try:
        os.makedirs(DST_MODEL_DIR)
    except OSError as e:
        if e.errno != 17:
            raise
    # Update the datatype of the specified variable from float32 to int8.
    var_name = "image"
    main_block = program.block(0)
    var = main_block.var(var_name)
    # core.VarDesc.VarType.UINT8
    # core.VarDesc.VarType.INT8
    # core.VarDesc.VarType.INT16
    # core.VarDesc.VarType.INT32
    # core.VarDesc.VarType.INT64
    # core.VarDesc.VarType.FP32
    # core.VarDesc.VarType.FP64
    # core.VarDesc.VarType.FP16
    # core.VarDesc.VarType.BF16
    # core.VarDesc.VarType.COMPLEX64
    # core.VarDesc.VarType.COMPLEX128
    # core.VarDesc.VarType.BOOL
    var.desc.set_dtype(core.VarDesc.VarType.INT8)
    print('--- new feed_target_names ---')
    print(feed_target_names)
    print('--- new fetch_targets ---')
    print(fetch_targets)
    if len(MODEL_FILE) == 0 and len(PARAMS_FILE) == 0:
        fluid.io.save_inference_model(DST_MODEL_DIR, feed_target_names,
                                      fetch_targets, exe, program)
    else:
        fluid.io.save_inference_model(
            DST_MODEL_DIR,
            feed_target_names,
            fetch_targets,
            exe,
            program,
            model_filename=MODEL_FILE,
            params_filename=PARAMS_FILE)
    print("Done.")


if __name__ == '__main__':
    main()
