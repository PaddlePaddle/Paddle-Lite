#!/bin/bash
#
# Start the CI task of examining Android inference lib compiling.
set +x
set -e

#####################################################################################################
# Usage: test the publish period on Android platform.
# Data: 20210104
# Author: DannyIsFunny
#####################################################################################################

#####################################################################################################
# 1. global variables, you can change them according to your requirements
#####################################################################################################
# Python version
PYTHON_VERSION=(2.7)
# Absolute path of Paddle-Lite source code.
SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
WORKSPACE=${SHELL_FOLDER%tools/ci_tools*}
# OpenCL
BUILD_OPENCL=ON
# Model download url
mobilenet_v1_url=http://paddle-inference-dist.bj.bcebos.com/mobilenet_v1.tar.gz

####################################################################################################
# Functions of downloading model file
# Arguments:
#   1. model_name
#   2. downloading url
####################################################################################################
function prepare_model {
  local model_name=$1
  local download_url=$2
  wget $download_url
  tar zxf $model_name.tar.gz
}

####################################################################################################
# Functions of Android compiling test.
# Globals:
#   WORKSPACE
# Arguments:
#   1. python version
####################################################################################################
function publish_inference_lib {
  cd $WORKSPACE
  # Local variables
  python_version=$1
  # Remove Compiling Cache
  rm -rf build*
  for python_version in ${PYTHON_VERSION[@]}; do
    # Step1. Compiling python installer on mac
    ./lite/tools/build.sh \
      --build_python=ON \
      --python_version=$python_version \
      --build_opencl=$BUILD_OPENCL \
      x86
    # Step2. Checking results: cplus and python inference lib.
    if [ -d build.lite.x86/inference_lite_lib/cxx/lib ] && [ -d build.lite.x86/inference_lite_lib/python/install/dist ]; then
      # test python installer
      cd build.lite.x86
      python$python_version -m pip install --force-reinstall  inference_lite_lib/python/install/dist/*.whl
      # download test model
      prepare_model mobilenet_v1 $mobilenet_v1_url
      # test opt
      paddle_lite_opt
      paddle_lite_opt --model_dir=mobilenet_v1 --optimize_out=mobilenet_v1_arm
      paddle_lite_opt --model_dir=mobilenet_v1 --valid_targets=x86 --optimize_out=mobilenet_v1_x86
      paddle_lite_opt --model_dir=mobilenet_v1 --valid_targets=x86_opencl --optimize_out=mobilenet_v1_x86_opencl
      # test inference demo
      cd inference_lite_lib/demo/python
      python$python_version mobilenetv1_full_api.py  --model_dir=$WORKSPACE/build.lite.x86/mobilenet_v1
      python$python_version mobilenetv1_light_api.py  --model_dir=$WORKSPACE/build.lite.x86/mobilenet_v1_x86.nb
      python$python_version mobilenetv1_light_api.py  --model_dir=$WORKSPACE/build.lite.x86/mobilenet_v1_x86_opencl.nb
      # uninstall
	  python$python_version -m pip uninstall -y paddlelite
    else
      # Error message.
      echo "**************************************************************************************"
      echo -e "* Mac python installer compiling task failed on the following instruction:"
      echo -e "*     ./lite/tools/build.sh --with_python=ON --python_version=$python_version"
      echo "**************************************************************************************"
      exit 1
    fi
  done
}

# Compiling test
for version in ${PYTHON_VERSION[@]}; do
    publish_inference_lib $version
done    
