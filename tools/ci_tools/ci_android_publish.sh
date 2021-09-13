#!/bin/bash
#
# Start the CI task of examining Android inference lib and benchmark tool compiling .
shopt -s expand_aliases
set +x
set -e

#####################################################################################################
# Usage: test the publish period on Android platform.
# Author: DannyIsFunny
#####################################################################################################

#####################################################################################################
# 1. global variables, you can change them according to your requirements
#####################################################################################################
# Architecture: armv7 or armv8, default armv8.
ARCH=(armv8 armv7)
# Toolchain: gcc or clang, default gcc.
TOOLCHAIN=(gcc clang)
# OpenCL
OPENCL=(ON OFF)
# Absolute path of Paddle-Lite source code.
SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
WORKSPACE=${SHELL_FOLDER%tools/ci_tools*}
# ModelS URL
MODELS_URL=( "http://paddle-inference-dist.bj.bcebos.com/mobilenet_v1.tar.gz" \  # mobilenet_v1
           )
# Directory on target
TARGET_DIR=/data/local/tmp/benchmark_ci_test/
# Device ID, just pick one available device
DEVICE_ID=$(adb devices | sed -n "2p" | awk '{print $1}')
alias ADB='adb -s '${DEVICE_ID}''
#####################################################################################################

source ${SHELL_FOLDER}/util.sh

####################################################################################################
# Functions of Android compiling test.
# Globals:
#   WORKSPACE
# Arguments:
#   1. architecture
#   2. toolchain
#   3. support opencl or not
#   4. with extra or not
# Returns:
#   None
####################################################################################################
function publish_inference_lib {
  local arch=$1
  local toolchain=$2
  local with_opencl=$3
  local with_extra=$4
  cd $WORKSPACE
  # Remove Compiling Cache
  rm -rf build*
  # Compiling inference library
  cmd="time ./lite/tools/build_android.sh --arch=$arch --toolchain=$toolchain --with_extra=$with_extra --with_opencl=$with_opencl --with_static_lib=ON"
  $cmd
  # Checking results: cplus and java inference lib.
  if [ -d build*/inference*/cxx/lib ] && [ -d build*/inference*/java/so ]; then
    cxx_results=$(ls build*/inference*/cxx/lib | wc -l)
    java_results=$(ls build*/inference*/java/so | wc -l)
      if [ $cxx_results -ge 2 ] && [ $java_results -ge 1 ]; then
          return 0
      fi
  fi
  # Error message.
  echo "**************************************************************************************"
  echo -e "* Android compiling task failed on the following instruction:"
  echo -e "*    ${cmd}"
  echo "**************************************************************************************"
  exit 1
}


function publish_benchmark_bin {
  local arch=$1
  local toolchain=$2
  local exe=benchmark_bin
  cd $WORKSPACE
  # Remove Compiling Cache
  rm -rf build*
  # Compiling inference library
  cmd="./lite/tools/build_android.sh --arch=$arch --toolchain=$toolchain --with_benchmark=ON full_publish"
  ${cmd}
  # Checking results
  if [ ! -f build*/lite/api/${exe} ]; then
    echo "**************************************************************************************"
    echo -e "* ${exe} is not exist!"
    echo -e "* Android compiling task failed on the following instruction:"
    echo -e "*     ${cmd}"
    echo "**************************************************************************************"
    exit 1
  fi
  # Run 
  if [ -z "$DEVICE_ID" ]; then
    echo "No devices found!"
    exit 1
  else
    local exe_file=$(ls build*/lite/api/${exe})
    ADB push ${exe_file} ${TARGET_DIR}
    ADB shell chmod +x ${TARGET_DIR}/${exe}
    for model in `ls ${model_dir}`; do
      for bd in arm opencl; do
        ADB shell ${TARGET_DIR}/${exe} \
            --uncombined_model_dir=${TARGET_DIR}/Models/${model} \
            --input_shape=1,3,224,224 \
            --backend=${bd} \
            --warmup=2 \
            --repeats=5
      done
    done
    ADB shell rm -rf ${TARGET_DIR}
  fi
}


# Download models
model_dir=${WORKSPACE}/Models
prepare_models ${model_dir}

# Push models to mobile device
ADB shell mkdir -p ${TARGET_DIR}
ADB push ${model_dir} ${TARGET_DIR}

# Compiling test: Not fully test for time-saving
for arch in armv8; do
  for toolchain in gcc; do
    for opencl in ${OPENCL[@]}; do
      cd $WORKSPACE
      publish_inference_lib $arch $toolchain $opencl ON
    done
    publish_benchmark_bin $arch $toolchain
  done
done

# Clear
rm -rf ${model_dir}
ADB shell rm -rf ${TARGET_DIR}
