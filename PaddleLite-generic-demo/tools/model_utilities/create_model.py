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

MODEL_DIR = "./output_model"
MODEL_FILE = "model.pdmodel"
PARAMS_FILE = "model.pdiparams"

#MODEL_FILE = ""
#PARAMS_FILE = ""


def main(argv=None):
    # Build network
    x = paddle.static.data(name='x', shape=[-1, 3, 80, 80, 2], dtype='float32')
    y = paddle.full(shape=[1], fill_value=2.0, dtype='float32')
    z = paddle.full(shape=[1], fill_value=2.0, dtype='float32')
    w = paddle.static.create_parameter(
        shape=[1, 3, 80, 80, 2],
        dtype='float32',
        attr=paddle.ParamAttr(
            initializer=paddle.nn.initializer.Assign(
                np.ones([1, 3, 80, 80, 2])),
            trainable=False))
    m = x * y
    n = paddle.pow(m, z)
    out = n * w
    # Set the input data to execute the network and print the output data
    program = paddle.static.default_main_program()
    exe = paddle.static.Executor(place=paddle.CPUPlace())
    x_data = np.ones(shape=[1, 3, 80, 80, 2], dtype=np.float32)
    exe.run(paddle.static.default_startup_program())
    [out_data] = exe.run(program,
                         feed={'x': x_data},
                         fetch_list=[out],
                         return_numpy=True)
    print(out_data.shape)
    #print(out_data)
    # Save the network to model for inference
    if len(MODEL_FILE) == 0 and len(PARAMS_FILE) == 0:
        fluid.io.save_inference_model(MODEL_DIR, ['x'], [out], exe, program)
    else:
        fluid.io.save_inference_model(
            MODEL_DIR, ['x'], [out],
            exe,
            program,
            model_filename=MODEL_FILE,
            params_filename=PARAMS_FILE)
    print("Done.")


if __name__ == '__main__':
    main()
