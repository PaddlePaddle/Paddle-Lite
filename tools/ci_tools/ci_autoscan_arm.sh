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
PYTHON_VERSION=3.9
# Absolute path of Paddle-Lite source code.
SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
WORKSPACE=${SHELL_FOLDER%tools/ci_tools*}
# OpenCL
BUILD_OPENCL=ON
# Metal
BUILD_METAL=ON
# Common options
BUILD_EXTRA=ON
WITH_EXCEPTION=OFF
TARGETS=(ARM OpenCL Metal)

# Model download url

####################################################################################################
# Functions of operate unit test
# Arguments:
#   1. python version
####################################################################################################
function auto_scan_test {
  target_name=$1

  cd $WORKSPACE/lite/tests/unittest_py/rpc_service
  sh start_rpc_server.sh

  cd $WORKSPACE/lite/tests/unittest_py/op/
  unittests=$(ls)
  for test in ${unittests[@]}; do
    if [[ "$test" =~ py$ ]];then
      python3.8 $test --target=$target_name
    fi
  done

  cd $WORKSPACE/lite/tests/unittest_py/pass/
  unittests=$(ls)
  for test in ${unittests[@]}; do
    if [[ "$test" =~ py$ ]];then
      python3.8 $test --target=$target_name
    fi
  done
}

####################################################################################################
# Functions of Android compiling test.
# Globals:
#   WORKSPACE
####################################################################################################
function compile_publish_inference_lib {
  cd $WORKSPACE

  # Remove Compiling Cache
  rm -rf build*

  # Step1. Compiling python installer on mac
  cmd_line="./lite/tools/build_macos.sh --with_python=ON --with_opencl=$BUILD_OPENCL --with_metal=$BUILD_METAL --with_arm82_fp16=ON --python_version=$PYTHON_VERSION arm64"
  $cmd_line
  # Step2. Checking results: cplus and python inference lib.
  build_dir=build.macos.armmacos.armv8.metal.opencl

  if [ -d ${build_dir}/inference_lite_lib.armmacos.armv8.opencl.metal/python/install/dist ]; then
    #install deps
    python$PYTHON_VERSION -m pip install --force-reinstall  ${build_dir}/inference_lite_lib.armmacos.armv8.opencl.metal/python/install/dist/*.whl
    python3.8 -m pip install -r ./lite/tests/unittest_py/requirements.txt
  else
    # Error message.
    echo "**************************************************************************************"
    echo -e "Compiling task failed on the following instruction:\n $cmd_line"
    echo "**************************************************************************************"
    exit 1
  fi
}

function run_test {
  target_name=$1
  # operate test
  auto_scan_test  $target_name
}

compile_publish_inference_lib
for target in ${TARGETS[@]}; do
  run_test $target
done

# uninstall paddlelite
python$PYTHON_VERSION -m pip uninstall -y paddlelite
echo "Success."
