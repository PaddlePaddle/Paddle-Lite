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
# Absolute path of Paddle-Lite source code.
SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
WORKSPACE=${SHELL_FOLDER%tools/ci_tools*}
# Common options
BUILD_EXTRA=ON
WITH_EXCEPTION=OFF
TARGETS=(Host, X86)
PYTHON_VERSION=3.7

# Model download url

####################################################################################################
# Functions of operate unit test
# Arguments:
#   1. python version
####################################################################################################
function auto_scan_test {
  target_name=$1

  cd $WORKSPACE/lite/tests/unittest_py/op/
  unittests=$(ls)
  for test in ${unittests[@]}; do
    if [[ "$test" =~ py$ ]];then
      python$PYTHON_VERSION $test --target=$target_name
    fi
  done

  cd $WORKSPACE/lite/tests/unittest_py/pass/
  unittests=$(ls)
  for test in ${unittests[@]}; do
    if [[ "$test" =~ py$ ]];then
      python$PYTHON_VERSION $test --target=$target_name
    fi
  done
}

####################################################################################################
# Functions of Android compiling test.
# Globals:
#   WORKSPACE
# Arguments:
#   1. target name
####################################################################################################
function publish_inference_lib {
  cd $WORKSPACE
  # Local variables
  target_name=$1
  # Remove Compiling Cache
  rm -rf build*

  # Step1. Compiling python installer on mac
  ./lite/tools/build.sh \
    --build_python=ON \
    --python_version=$PYTHON_VERSION \
    --build_extra=$BUILD_EXTRA \
    x86
  # Step2. Checking results: cplus and python inference lib.
  build_dir=build.lite.x86

  if [ -d ${build_dir}/inference_lite_lib/python/install/dist ]; then
    # install deps
    python$PYTHON_VERSION -m pip install --force-reinstall  ${build_dir}/inference_lite_lib/python/install/dist/*.whl
    python$PYTHON_VERSION -m pip install -r ./lite/tests/unittest_py/requirements.txt
    # operate test
    auto_scan_test  $target_name
    # uninstall paddlelite
    python$PYTHON_VERSION -m pip uninstall -y paddlelite
    echo "Success."
  else
    # Error message.
    echo "**************************************************************************************"
    echo -e "* Python installer compiling task failed on the following instruction:"
    echo -e "*     ./lite/tools/build.sh --with_python=ON --python_version=$PYTHON_VERSION
    --build_extra=$BUILD_EXTRA x86"
    echo "**************************************************************************************"
    exit 1
  fi
}

for target in ${TARGETS[@]}; do
  publish_inference_lib $target
done
