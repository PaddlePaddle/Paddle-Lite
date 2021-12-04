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
PYTHON_VERSION=(3.7)
# Absolute path of Paddle-Lite source code.
SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
WORKSPACE=${SHELL_FOLDER%tools/ci_tools*}
# OpenCL
BUILD_OPENCL=OFF
# Common options
BUILD_EXTRA=ON
WITH_EXCEPTION=OFF

SkipHostTest=(test_beam_serach_op.py test_beam_serach_decode_op.py test_box_clip_op.py)
SkipX86Test=(test_batch_nom_op.py test_bilinear_interp_op.py test_bilinear_interp_v2_op.py test_conv2d_op.py test_gru_op.py test_sequence_pool_op.py test_sequence_topk_avg_pooling_op.py)
SkipOpenclTest=()
# Model download url

####################################################################################################
# Functions of operate unit test
# Arguments:
#   1. python version
####################################################################################################
function auto_scan_test {
  python_version=$1

  if [ ${BUILD_OPENCL} = ON ];then
    cd $WORKSPACE/lite/tests/unittest_py/op/backends/opencl
    unittests=$(ls)
    for test in ${unittests[@]}; do
      if [[ "${SkipOpenclTest[@]}" =~ "$test" ]];then
        continue
      else
        python$python_version $test
      fi
    done
  else
    cd $WORKSPACE/lite/tests/unittest_py/op/backends/host
    unittests=$(ls)
    for test in ${unittests[@]}; do
      if [[ "${SkipHostTest[@]}" =~ "$test" ]];then
        continue
      else
        python$python_version $test
      fi
    done

    cd $WORKSPACE/lite/tests/unittest_py/op/backends/x86
    unittests=$(ls)
    for test in ${unittests[@]}; do
      if [[ "${SkipX86Test[@]}" =~ "$test" ]];then
        continue
      else
        python$python_version $test
      fi
  done
  fi
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
      --build_extra=$BUILD_EXTRA \
      x86
    # Step2. Checking results: cplus and python inference lib.
    build_dir=build.lite.x86
    if [ ${BUILD_OPENCL} = ON ]; then
      build_dir=build.lite.x86.opencl
    fi

    if [ -d ${build_dir}/inference_lite_lib/python/install/dist ]; then
      #install deps
      python$python_version -m pip install --force-reinstall  ${build_dir}/inference_lite_lib/python/install/dist/*.whl
      python$python_version -m pip install -r ./lite/tests/unittest_py/requirements.txt
      #operate test
      auto_scan_test $python_version
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

function main() {
  # Parse command line.
  for i in "$@"; do
    case $i in
    --BUILD_OPENCL=*)
      BUILD_OPENCL="${i#*=}"
      shift
      ;;
    *)
      echo "Unknown option, exit"
      exit 1
      ;;
    esac
  done
}

main $@
# Compiling test
for version in ${PYTHON_VERSION[@]}; do
    publish_inference_lib $version
done
