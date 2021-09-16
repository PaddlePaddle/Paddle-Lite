#!/bin/bash
shopt -s expand_aliases
set -ex

# Global variables
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
# Fast fail if no devices found
if [[ -z "$DEVICE_ID" ]]; then
    echo -e "\e[1;31mNo device found!\e[0m"
    exit 1
fi
alias ADB='adb -s '$DEVICE_ID''

# helper functions
source ${SHELL_FOLDER}/utils.sh

# if operating in mac env, we should expand the maximum file num
os_name=$(uname -s)
if [ ${os_name} == "Darwin" ]; then
  ulimit -n 1024
fi

function build_and_test_benchmark {
  local os=$1
  local arch=$2
  local toolchain=$3
  local model_dir=$4
  local exe=benchmark_bin
  local cmd_line
  cd $WORKSPACE

  # Remove Compiling Cache
  rm -rf build.*

  # Compiling
  cmd_line="./lite/tools/build_${os}.sh --arch=$arch --toolchain=$toolchain --with_benchmark=ON full_publish"
  ${cmd_line}

  # Checking results
  local exe_file=$(ls build.*/lite/api/${exe})
  if [ ! -f $exe_file ]; then
    echo -e "\e[1;31m $exe_file is not exist! \e[0m"
    echo -e "Android compiling task failed on the following instruction:\n $cmd"
    exit 1
  fi

  # Run
  if [[ "$os" == "android" ]]; then
    if [[ -z "$DEVICE_ID" ]]; then
      echo -e "\e[1;31m No device found! \e[0m"
      exit 1
    else
      ADB push $exe_file $TARGET_DIR
      ADB shell chmod +x ${TARGET_DIR}/${exe}
      for model in `ls $model_dir`; do
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
  elif [[ "$os" == "linux" ]]; then
    if [[ "$arch" == "x86" ]]; then
      local mklml_so_name="libmklml_intel.so"
      local mklml_so_path=$(find ./build.* -name $mklml_so_name | head -n 1)
      if [[ -z "$mklml_so_path" ]]; then
        echo -e "\e[1;31m mklml.so not found! \e[0m"
        exit 1
      fi
      mklml_so_path=$(dirname $mklml_so_path)
      export LD_LIBRARY_PATH=${mklml_so_path}:$LD_LIBRARY_PATH
      for model in `ls $model_dir`; do
        $exe_file \
            --uncombined_model_dir=./Models/${model} \
            --input_shape=1,3,224,224 \
            --backend=x86 \
            --warmup=2 \
            --repeats=5
      done
    else # arm
      # TODO:
      # copy $exe_file & $model_dir to target device
      # Run
    fi
  fi
}

function main() {
  # Download models
  local model_dir=${WORKSPACE}/Models
  prepare_models $model_dir

  # Push models to mobile device
  ADB shell mkdir -p $TARGET_DIR
  ADB push $model_dir $TARGET_DIR

  # Compiling test: Not fully test for time-saving
  local device_path=$1
  local os="android"
  for arch in armv8; do
    for toolchain in clang; do
      build_and_test_benchmark $os $arch $toolchain $model_dir
    done
  done

  os="linux"
  for arch in x86 armv8; do
    for toolchain in gcc; do
      build_and_test_benchmark $os $arch $toolchain $model_dir
    done
  done
}

main $@
