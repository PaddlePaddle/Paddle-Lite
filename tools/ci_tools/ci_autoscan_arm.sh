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
PYTHON_VERSION=(3.9)
# Absolute path of Paddle-Lite source code.
SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
WORKSPACE=${SHELL_FOLDER%tools/ci_tools*}
# OpenCL
BUILD_OPENCL=OFF
# Common options
BUILD_EXTRA=ON
WITH_EXCEPTION=OFF

# Model download url

####################################################################################################
# Functions of operate unit test
# Arguments:
#   1. python version
####################################################################################################
function auto_scan_test {
  python_version=$1
  cd $WORKSPACE/lite/tests/unittest_py/rpc_service
  sh start_rpc_server.sh
  cd $WORKSPACE/lite/tests/unittest_py/arm
  unittests=$(ls)
  for test in ${unittests[@]}; do
    python$python_version $test
  done
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
    ./lite/tools/build_macos.sh --with_python=ON --python_version=3.9 arm64
    # Step2. Checking results: cplus and python inference lib.
    build_dir=build.macos.armmacos.armv8
#    if [ ${BUILD_OPENCL} = ON ]; then
#      build_dir=build.lite.x86.opencl
#    fi

    if [ -d ${build_dir}/inference_lite_lib.armmacos.armv8/python/install/dist ]; then
      #install deps
      python$python_version -m pip install --force-reinstall  ${build_dir}/inference_lite_lib.armmacos.armv8/python/install/dist/*.whl
      python3.8 -m pip install -r ./lite/tests/unittest_py/requirements.txt
      #operate test
      auto_scan_test 3.8
      # uninstall paddlelite
      python$python_version -m pip uninstall -y paddlelite
      echo "Success."
    else
      # Error message.
      echo "**************************************************************************************"
      echo -e "* Python installer compiling task failed on the following instruction:"
      echo -e "*     ./lite/tools/build.sh --with_python=ON --python_version=$python_version
      --build_opencl=$BUILD_OPENCL --build_extra=$BUILD_EXTRA x86"
      echo "**************************************************************************************"
      exit 1
    fi
  done
}

# Compiling test
for version in ${PYTHON_VERSION[@]}; do
    publish_inference_lib $version
done
