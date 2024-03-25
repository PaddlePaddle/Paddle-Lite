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
import cv2
import numpy as np
import paddle
import paddle.fluid as fluid
from paddle.fluid import core

paddle.enable_static()

MODEL_NAME = "conv_bn_relu_224_fp32"
MODEL_FILE = ""
PARAMS_FILE = ""


def main(argv=None):
    # Load model
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    model_dir = "../../assets/models/" + MODEL_NAME
    if len(MODEL_FILE) == 0 and len(PARAMS_FILE) == 0:
        [program, feed_target_names,
         fetch_targets] = fluid.io.load_inference_model(model_dir, exe)
    else:
        [program, feed_target_names,
         fetch_targets] = fluid.io.load_inference_model(
             model_dir,
             exe,
             model_filename=MODEL_FILE,
             params_filename=PARAMS_FILE)
    print("--- feed_target_names ---")
    print(feed_target_names)
    print("--- fetch_targets ---")
    print(fetch_targets)
    # Preprocess
    input_tensors = {'image': np.ones([1, 3, 224, 224]).astype(np.float32)}
    # Inference
    output_tensors = exe.run(program=program,
                             feed=input_tensors,
                             fetch_list=fetch_targets,
                             return_numpy=False)
    # Postprocess
    for output_tensor in output_tensors:
        output_data = np.array(output_tensor)
        print(output_data.shape)
        print(output_data)
    print("Done.")


if __name__ == '__main__':
    main()
