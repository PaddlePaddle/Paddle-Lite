# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
'''
Paddle-Lite light python api demo
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

from paddlelite.lite import *
import numpy as np

# Command arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_dir", default="", type=str, help="Non-combined Model dir path")
parser.add_argument(
    "--input_shape",
    default=[1, 3, 224, 224],
    nargs='+',
    type=int,
    required=False,
    help="Model input shape, eg: 1 3 224 224. Defalut: 1 3 224 224")
parser.add_argument(
    "--backend",
    default="",
    type=str,
    help="To use a particular backend for execution. Should be one of: arm|opencl|x86|x86_opencl|metal"
)
parser.add_argument(
    "--image_path", default="", type=str, help="The path of test image file")
parser.add_argument(
    "--label_path", default="", type=str, help="The path of label file")
parser.add_argument(
    "--print_results",
    type=bool,
    default=False,
    help="Print results. Default: False")


def RunModel(args):
    # 1. Set config information
    config = MobileConfig()
    config.set_model_from_file(args.model_dir)

    if args.backend.upper() in ["OPENCL", "X86_OPENCL"]:
        bin_path = "./"
        bin_name = "lite_opencl_kernel.bin"
        config.set_opencl_binary_path_name(bin_path, bin_name)
        '''
        opencl tune option:
        CL_TUNE_NONE
        CL_TUNE_RAPID
        CL_TUNE_NORMAL
        CL_TUNE_EXHAUSTIVE
        '''
        tuned_path = "./"
        tuned_name = "lite_opencl_tuned.bin"
        config.set_opencl_tune(CLTuneMode.CL_TUNE_NORMAL, tuned_path,
                               tuned_name, 4)
        '''
        opencl precision option:
        CL_PRECISION_AUTO, first fp16 if valid, default
        CL_PRECISION_FP32, force fp32
        CL_PRECISION_FP16, force fp16
        '''
        config.set_opencl_precision(CLPrecisionType.CL_PRECISION_AUTO)
    elif args.backend.upper() in ["METAL"]:
        # set metallib path
        import paddlelite, os
        module_path = os.path.dirname(paddlelite.__file__)
        config.set_metal_lib_path(module_path + "/libs/lite.metallib")
        config.set_metal_use_mps(True)

    # 2. Create paddle predictor
    predictor = create_paddle_predictor(config)

    # 3. Set input data
    input_tensor = predictor.get_input(0)
    c, h, w = args.input_shape[1], args.input_shape[2], args.input_shape[3]
    read_image = len(args.image_path) != 0 and len(args.label_path) != 0
    if read_image == True:
        import cv2
        with open(args.label_path, "r") as f:
            label_list = f.readlines()
        image_mean = [0.485, 0.456, 0.406]
        image_std = [0.229, 0.224, 0.225]
        image_data = cv2.imread(args.image_path)
        image_data = cv2.resize(image_data, (h, w))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
        image_data = image_data.transpose((2, 0, 1)) / 255.0
        image_data = (image_data - np.array(image_mean).reshape(
            (3, 1, 1))) / np.array(image_std).reshape((3, 1, 1))
        image_data = image_data.reshape([1, c, h, w]).astype('float32')
        input_tensor.from_numpy(image_data)
    else:
        input_tensor.from_numpy(np.ones((1, c, h, w)).astype("float32"))

    # 4. Run model
    predictor.run()

    # 5. Get output data
    output_tensor = predictor.get_output(0)
    output_data = output_tensor.numpy()
    if args.print_results == True:
        print("result data:\n{}".format(output_data))
    print("mean:{:.6e}, std:{:.6e}, min:{:.6e}, max:{:.6e}".format(
        np.mean(output_data),
        np.std(output_data), np.min(output_data), np.max(output_data)))

    # 6. Post-process
    if read_image == True:
        output_data = output_data.flatten()
        class_id = np.argmax(output_data)
        class_name = label_list[class_id]
        score = output_data[class_id]
        print("class_name: {} score: {}".format(class_name, score))


if __name__ == '__main__':
    args = parser.parse_args()
    RunModel(args)
