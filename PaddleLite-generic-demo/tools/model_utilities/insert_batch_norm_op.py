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
    input_name = "image"
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    # Find an op whose output is the specified variable, then insert a print op after it.
    output_name = input_name + "_bn_out"
    scope = fluid.global_scope()
    main_block = program.block(0)
    input_var = main_block.var(input_name)
    channel_size = input_var.shape[1]
    output_var = main_block.create_var(
        name=output_name,
        shape=input_var.shape,
        dtype=input_var.dtype,
        type=input_var.type,
        persistable=False,
        stop_gradient=False)
    # bias
    bn_bias_name = input_name + "_bn_bias"
    bn_bias_var = main_block.create_var(
        name=bn_bias_name,
        shape=[channel_size],
        dtype=input_var.dtype,
        type=input_var.type,
        persistable=True,
        stop_gradient=False)
    scope.var(bn_bias_name).get_tensor().set(np.zeros(
        [channel_size], dtype=np.float32),
                                             place)
    # mean
    bn_mean_name = input_name + "_bn_mean"
    bn_mean_var = main_block.create_var(
        name=bn_mean_name,
        shape=[channel_size],
        dtype=input_var.dtype,
        type=input_var.type,
        persistable=True,
        stop_gradient=False)
    scope.var(bn_mean_name).get_tensor().set(mean, place)
    # scale
    bn_scale_name = input_name + "_bn_scale"
    bn_scale_var = main_block.create_var(
        name=bn_scale_name,
        shape=[channel_size],
        dtype=input_var.dtype,
        type=input_var.type,
        persistable=True,
        stop_gradient=False)
    scope.var(bn_scale_name).get_tensor().set(np.ones(
        [channel_size], dtype=np.float32) / std,
                                              place)
    # variance
    bn_variance_name = input_name + "_bn_variance"
    bn_variance_var = main_block.create_var(
        name=bn_variance_name,
        shape=[channel_size],
        dtype=input_var.dtype,
        type=input_var.type,
        persistable=True,
        stop_gradient=False)
    scope.var(bn_variance_name).get_tensor().set(np.ones(
        [channel_size], dtype=np.float32),
                                                 place)
    # saved_mean
    bn_saved_mean_name = input_name + "_bn_saved_mean"
    bn_saved_mean_var = main_block.create_var(
        name=bn_saved_mean_name,
        shape=[channel_size],
        dtype=input_var.dtype,
        type=input_var.type,
        persistable=False,
        stop_gradient=False)
    # saved_variance
    bn_saved_variance_name = input_name + "_bn_saved_variance"
    bn_saved_variance_var = main_block.create_var(
        name=bn_saved_variance_name,
        shape=[channel_size],
        dtype=input_var.dtype,
        type=input_var.type,
        persistable=False,
        stop_gradient=False)
    main_block._insert_op(
        index=1,
        type='batch_norm',
        inputs={
            'X': input_var,
            'Bias': bn_bias_var,
            'Mean': bn_mean_var,
            'Scale': bn_scale_var,
            'Variance': bn_variance_var
        },
        outputs={
            'Y': output_var,
            'MeanOut': bn_mean_var,
            'SavedMean': bn_saved_mean_var,
            'SavedVariance': bn_saved_variance_var,
            'VarianceOut': bn_variance_var
        },
        attrs={
            'epsilon': 0.000009999999747378752,
            'fuse_with_relu': False,
            'momentum': 0.8999999761581421,
            'use_global_stats': False,
        })
    for i in range(len(main_block.ops)):
        op_desc = main_block.ops[i].desc
        op_type = op_desc.type()
        if op_type == 'batch_norm':
            continue
        for arg_name in op_desc.inputs():
            input_names = op_desc.input(arg_name)
            for j in range(len(input_names)):
                if input_names[j] == input_name:
                    print("Found op %s!" % op_type)
                    found = True
                    input_names[j] = output_name
                    op_desc.set_input(arg_name, input_names)
                    # append_op should not be used as it may be incorrectly pruned.
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
