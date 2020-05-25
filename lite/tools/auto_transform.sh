#!/usr/bin/env bash
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
# ==============================================================================

#set -u  # Check for undefined variables

# Global variables
###########################################
# (1) x2paddle variables
framework="caffe"        # framework=(caffe|tensorflow|onnx)
prototxt=""
weight=""
model=""
# fluid_save__dir: the path of x2paddlei's output result; this is used as `model_dir` of opt
fluid_save_dir="saved_fluid"
###########################################
# (2)opt variables
valid_targets="arm"       # valid_targets=(arm|opencl|x86|npu|xpu)
optimize_out="lite_opt_dir"


# check current system
system=`uname -s`
opt=""
if [ ${system} == "Darwin" ]; then
  opt=opt_mac
else
  opt=opt
fi

function check_x2paddle {
  message=$(which x2paddle)
  if [ ! $message ]; then 
    echo "please install x2paddle environment first, you can install it according to https://github.com/PaddlePaddle/X2Paddle#%E7%8E%AF%E5%A2%83%E4%BE%9D%E8%B5%96"
    exit 1
  fi
}
function check_model_optimize_tool {
  has_opt=$(find $opt)
  if [ -z "$has_opt" ]; then
    wget https://paddlelite-data.bj.bcebos.com/model_optimize_tool/$opt
    chmod +x $opt
  fi
}
function x2paddle_transform {
  check_x2paddle
  x2paddle 
  if [ "$framework" == "caffe" ]; then
    x2paddle --framework caffe \
            --prototxt=$prototxt \
      	    --weight=$weight \
            --save_dir=$fluid_save_dir
  elif [ "$framework" == "tensorflow" ]; then
    x2paddle --framework=tensorflow \
 	     --model=$model \
             --save_dir=$fluid_save_dir
  elif [ "$framework" == "onnx" ]; then
    x2paddle --framework=onnx \
             --model=$model \
             --save_dir=$fluid_save_dir
  else
    echo "error: unsupported framwork, x2paddle supports three framework: caffe、tensorflow and onnx."
    exit 1
  fi
}

function model_optimimize_tool_transform {
     check_model_optimize_tool
     ./$opt \
       --model_dir=$fluid_save_dir/inference_model \
       --optimize_out_type=naive_buffer \
       --optimize_out=$optimize_out \
       --valid_targets=$valid_targets 
}

function print_usage {
    set +x
    echo "\nUSAGE:"
    echo "    auto_build.sh combines the function of x2paddle and opt, it can "
    echo "    tranform model from tensorflow/caffe/onnx form into paddle-lite naive-buffer form."
    echo "----------------------------------------"
    echo "example:"
    echo "    ./auto_build.sh --framework=tensorflow --model=tf_model.pb --optimize_out=opt_model_result"
    echo "----------------------------------------"
    echo  "Arguments about x2paddle:"
    echo "    --framework=(tensorflow|caffe|onnx);"
    echo "    --model='model file for tensorflow or onnx';"
    echo "    --prototxt='proto file for caffe' --weight='weight file for caffe'"

    echo "For TensorFlow:"
    echo "   --framework=tensorflow --model=tf_model.pb"
    echo
    echo "For Caffe:"
    echo "   --framework=caffe --prototxt=deploy.prototxt --weight=deploy.caffemodel"
    echo
    echo "For ONNX"
    echo "   --framework=onnx --model=onnx_model.onnx"
    echo
    echo "Arguments about opt:"
    echo "    --valid_targets=(arm|opencl|x86|npu|xpu); valid targets on Paddle-Lite."
    echo "    --fluid_save_dir='path to outputed model after x2paddle'"
    echo "    --optimize_out='path to outputed Paddle-Lite model'"
    echo "----------------------------------------"
    echo
}

function main {
    # Parse command line.
    if [ $# -eq 0 ] ; then
       print_usage
       exit 1
    fi
    for i in "$@"; do
        case $i in
            --framework=*)
                framework="${i#*=}"
                shift
                ;;
            --prototxt=*)
                prototxt="${i#*=}"
                shift
                ;;
            --weight=*)
                weight="${i#*=}"
                shift
                ;;
            --model=*)
                model="${i#*=}"
                shift
                ;;
            --fluid_save_dir=*)
                fluid_save_dir="${i#*=}"
                shift
                ;;
            --valid_targets=*)
                valid_targets="${i#*=}"
                shift
                ;;
            --optimize_out=*)
                optimize_out="${i#*=}"
                shift
                ;;
            *)
                # unknown option
                print_usage
                exit 1
                ;;
        esac
    done
    x2paddle_transform
    model_optimimize_tool_transform
}

main $@
