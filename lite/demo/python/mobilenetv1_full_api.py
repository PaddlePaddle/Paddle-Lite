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
Paddle-Lite full python api demo
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
parser.add_argument("--model_file", default="", type=str, help="Model file")
parser.add_argument(
    "--param_file", default="", type=str, help="Combined model param file")
parser.add_argument(
    "--enable_opencl", action="store_true", help="Enable OpenCL or not")
parser.add_argument(
    "--use_metal",
    action="store_true",
    default=False,
    help="use metal on Appel GPU. Default: False")
parser.add_argument(
    "--disable_print_results",
    action="store_false",
    help="Print results or not")


def RunModel(args):
    # 1. Set config information
    config = CxxConfig()
    if args.model_file != '' and args.param_file != '':
        config.set_model_file(args.model_file)
        config.set_param_file(args.param_file)
    else:
        config.set_model_dir(args.model_dir)

    # For arm platform (armlinux), you can set places = [Place(TargetType.ARM, PrecisionType.FP32)]

    if args.enable_opencl:
        places = [
            Place(TargetType.OpenCL, PrecisionType.FP16,
                  DataLayoutType.ImageDefault), Place(
                      TargetType.OpenCL, PrecisionType.FP16,
                      DataLayoutType.ImageFolder),
            Place(TargetType.OpenCL, PrecisionType.FP32, DataLayoutType.NCHW),
            Place(TargetType.OpenCL, PrecisionType.Any,
                  DataLayoutType.ImageDefault), Place(
                      TargetType.OpenCL, PrecisionType.Any,
                      DataLayoutType.ImageFolder),
            Place(TargetType.OpenCL, PrecisionType.Any, DataLayoutType.NCHW),
            Place(TargetType.X86, PrecisionType.FP32),
            Place(TargetType.ARM, PrecisionType.FP32),
            Place(TargetType.Host, PrecisionType.FP32)
        ]
        '''
        Set opencl kernel binary.
        Large addtitional prepare time is cost due to algorithm selecting and
        building kernel from source code.
        Prepare time can be reduced dramitically after building algorithm file
        and OpenCL kernel binary on the first running.
        The 1st running time will be a bit longer due to the compiling time if
        you don't call `set_opencl_binary_path_name` explicitly.
        So call `set_opencl_binary_path_name` explicitly is strongly
        recommended.

        Make sure you have write permission of the binary path.
        We strongly recommend each model has a unique binary name.
        '''
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
    elif args.use_metal:
        # set metallib path
        import paddlelite, os
        module_path = os.path.dirname(paddlelite.__file__)
        config.set_metal_lib_path(module_path + "/libs/lite.metallib")
        config.set_metal_use_mps(True)
        # set places for Metal
        places = [
            Place(TargetType.Metal, PrecisionType.FP32,
                  DataLayoutType.MetalTexture2DArray), Place(
                      TargetType.Metal, PrecisionType.FP16,
                      DataLayoutType.MetalTexture2DArray),
            Place(TargetType.X86, PrecisionType.FP32),
            Place(TargetType.ARM, PrecisionType.FP32),
            Place(TargetType.Host, PrecisionType.FP32)
        ]
    else:
        places = [Place(TargetType.X86, PrecisionType.FP32)]

    config.set_valid_places(places)

    # 2. Create paddle predictor
    predictor = create_paddle_predictor(config)

    # 3. Set input data
    input_tensor = predictor.get_input(0)
    input_tensor.from_numpy(np.ones((1, 3, 224, 224)).astype("float32"))

    # 4. Run model
    predictor.run()

    # 5. Get output data
    output_tensor = predictor.get_output(0)
    output_data = output_tensor.numpy()
    if args.disable_print_results:
        print(output_data)


if __name__ == '__main__':
    args = parser.parse_args()
    RunModel(args)
