#!/bin/bash
shopt -s expand_aliases
set -ex

# Global variables
# Absolute path of Paddle-Lite source code.
SHELL_FOLDER=$(
  cd "$(dirname "$0")"
  pwd
)
WORKSPACE=${SHELL_FOLDER%tools/ci_tools*}
# ModelS URL
MODELS_URL=("http://paddle-inference-dist.bj.bcebos.com/mobilenet_v1.tar.gz" \  # mobilenet_v1
)
# Model zoo path
MODEL_DIR=${WORKSPACE}/Models
# The list of os for building(android,armlinux), such as "android"
OS_LIST="android"
# The list of arch abi for building(armv8,armv7,armv7hf), such as "armv8,armv7"
# for android devices, "armv8" for RK3399, "armv7hf" for Raspberry pi 3B
ARCH_LIST="armv8,armv7"
# The list of toolchains for building(gcc,clang), such as "gcc,clang"
# for android, "gcc" for armlinx
TOOLCHAIN_LIST="gcc,clang"
# Remote device type(0: adb, 1: ssh) for real android and armlinux devices
REMOTE_DEVICE_TYPE=0
# The list of the device names for the real android devices, use commas to separate them, such as "2GX0119401000796,0123456789ABCDEF,5aba5d0ace0b89f6"
# The list of the device infos for the real armlinux devices, its format is "dev0_ip_addr,dev0_port,dev0_usr_id,dev0_usr_pwd:dev1_ip_addr,dev0_port,dev1_usr_id,dev1_usr_pwd"
REMOTE_DEVICE_LIST="2GX0119401000796,0123456789ABCDEF"
# Work directory of the remote devices for running
REMOTE_DEVICE_WORK_DIR="/data/local/tmp/benchmark_ci_test/"

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
      for model in $(ls $model_dir); do
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
      for model in $(ls $model_dir); do
        $exe_file \
          --uncombined_model_dir=./Models/${model} \
          --input_shape=1,3,224,224 \
          --backend=x86 \
          --warmup=2 \
          --repeats=5
      done
    else # arm
      echo "skip armlinux"
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

function android_build_target() {
  local os=$1
  local arch=$2
  local toolchain=$3
  local extra_arguments=$4

  # Remove Compiling Cache
  rm -rf build.*

  # Compiling
  cmd_line="./lite/tools/build_android.sh --arch=$arch --toolchain=$toolchain --with_benchmark=ON full_publish"
  ${cmd_line}

  # Checking results
  local exe_file=$(ls build.*/lite/api/${exe})
  if [ ! -f $exe_file ]; then
    echo -e "\e[1;31m $exe_file is not exist! \e[0m"
    echo -e "Compiling task failed on the following instruction:\n $cmd_line"
    exit 1
  fi
}

function run_on_remote_device() {
  local remote_device_name=""
  local remote_device_work_dir=""
  local remote_device_check=""
  local remote_device_run=""
  local target_name=""
  local model_dir=""

  # Extract arguments from command line
  for i in "$@"; do
    case $i in
    --remote_device_name=*)
      remote_device_name="${i#*=}"
      shift
      ;;
    --remote_device_work_dir=*)
      remote_device_work_dir="${i#*=}"
      shift
      ;;
    --remote_device_check=*)
      remote_device_check="${i#*=}"
      shift
      ;;
    --remote_device_run=*)
      remote_device_run="${i#*=}"
      shift
      ;;
    --target_name=*)
      target_name="${i#*=}"
      shift
      ;;
    --model_dir=*)
      model_dir="${i#*=}"
      shift
      ;;
    *)
      shift
      ;;
    esac
  done

  # Copy the executable to the remote device
  local target_path=$(find ./build.* -name $target_name)
  if [[ -z "$target_path" ]]; then
    echo "$target_name not found!"
    exit 1
  fi
  $remote_device_run $remote_device_name shell "rm -f $remote_device_work_dir/$target_name"
  $remote_device_run $remote_device_name push "$target_path" "$remote_device_work_dir"

  local backends=""
  if [[ "$os" == "android" || "$os" == "armlinux" ]]; then
    backends="arm opencl,arm"
  else if [ "$os" == "linux" ]; then
    backends="x86"
  else if [ "$os" == "osx" ]; then
    backends="x86 opencl,x86"
  fi
  local res_file="result.txt"
  local command_line=""

  for backend in backends; do
    command_line="
                $target_name \
                --model_file=$model_dir/inference.pdmodel \
                --param_file=$model_dir/inference.pdiparams \
                --input_shape=1,3,224,224 \
                --backend=$backend \
                --warmup=10 \
                --repeats=50 \
                --result_path=$res_file
                "

    # Run the model on the remote device
    $remote_device_run $remote_device_name shell "cd $remote_device_work_dir; LD_LIBRARY_PATH=$LD_LIBRARY_PATH:.; rm -rf $res_file; $command_line; cd -"
    $remote_device_run $remote_device_name pull "${remote_device_work_dir}/${res_file}" .

    local avg_time=$(grep $res_file | awk '{print $3}' | tail -1)
    local avg_time_baseline=`jq .model[0].backends[0].arm_abi[0].device_id[0].avg_time_baseline tools/ci_tools/ci_benchmark_config.json`
    local avg_time_thres=$(echo "${avg_time_baseline}*${avg_time_thres_scale}" | bc)
    if [ 1 -eq "$(echo "${avg_time} > ${avg_time_thres}" | bc)" ]; then
      echo "avg_time: $avg_time > avg_time_thres: $avg_time_thres on device $remote_device_name !\nThis PR may reduce performace. Reject this PR."
    fi
  done
}

function build_and_test_on_remote_device() {
  local os_list=$1
  local arch_list=$2
  local toolchain_list=$3
  local build_target_func=$4
  local prepare_device_func=$5
  local remote_device_type=$6
  local remote_device_list=$7
  local remote_device_work_dir=$8
  local avg_time_baseline=$9
  local avg_time_thres_scale=${10}
  local model_dir=${11}
  local extra_arguments=${12}

  # Set helper functions to access the remote devices
  local remote_device_pick=ssh_device_pick
  local remote_device_check=ssh_device_check
  local remote_device_run=ssh_device_run
  if [[ $remote_device_type -eq 0 ]]; then
    remote_device_pick=adb_device_pick
    remote_device_check=adb_device_check
    remote_device_run=adb_device_run
  fi

  # Check remote devices: Fast fail if any device is not available
  local remote_device_names=$($remote_device_pick $remote_device_list)
  if [[ -z $remote_device_names ]]; then
    echo "No remote device available!"
    exit 1
  else
    echo "Found remote device(s): $remote_device_names."
  fi

  cd $WORKSPACE

  # Download models to host machine
  prepare_models $model_dir

  # Prepare device environment for running, such as push models to remote device, only once for one device
  $prepare_device_func $os $arch $toolchain $remote_device_name $remote_device_work_dir $remote_device_check $remote_device_run $model_dir

  # Run
  local oss=(${os_list//,/ })
  local archs=(${arch_list//,/ })
  local toolchains=(${toolchain_list//,/ })
  for os in $oss; do
    for arch in $archs; do
      for toolchain in $toolchains; do
        # Build
        echo "Build with $os+$arch+$toolchain ..."
        $build_target_func $os $arch $toolchain
        # Loop all remote devices
        for remote_device_name in $remote_device_names; do
          # Run
          run_on_remote_device --remote_device_name=$remote_device_name \
            --remote_device_work_dir=$remote_device_work_dir \
            --remote_device_check=$remote_device_check \
            --remote_device_run=$remote_device_run \
            --target_name="benchmark_bin" \
            --model_dir=$(basename $model_dir)
        done
        cd - >/dev/null
      done
    done
  done
}

function android_build_and_test() {
  build_and_test_on_remote_device $OS_LIST $ARCH_LIST $TOOLCHAIN_LIST \
      android_build_target android_prepare_device \
      $REMOTE_DEVICE_TYPE $REMOTE_DEVICE_LIST $REMOTE_DEVICE_WORK_DIR \
      $AVG_TIME_BASELINE $AVG_TIME_THRES_SCALE $MODEL_DIR
}

function main() {
  # Parse command line.
  for i in "$@"; do
    case $i in
    --os_list=*)
      OS_LIST="${i#*=}"
      shift
      ;;
    --arch_list=*)
      ARCH_LIST="${i#*=}"
      shift
      ;;
    --toolchain_list=*)
      TOOLCHAIN_LIST="${i#*=}"
      shift
      ;;
    --remote_device_type=*)
      REMOTE_DEVICE_TYPE="${i#*=}"
      shift
      ;;
    --remote_device_list=*)
      REMOTE_DEVICE_LIST="${i#*=}"
      shift
      ;;
    --remote_device_work_dir=*)
      REMOTE_DEVICE_WORK_DIR="${i#*=}"
      shift
      ;;
    --avg_time_baseline=*)
      AVG_TIME_BASELINE="${i#*=}"
      shift
      ;;
    --avg_time_thres_scale=*)
      AVG_TIME_THRES_SCALE="${i#*=}"
      shift
      ;;
    --model_dir=*)
      MODEL_DIR="${i#*=}"
      shift
      ;;
    android_build_and_test)
      android_build_and_test
      shift
      ;;
    armlinux_build_and_test)
      armlinux_build_and_test
      shift
      ;;
    linux_build_and_test)
      linux_build_and_test
      shift
      ;;
    *)
      # unknown option
      print_usage
      exit 1
      ;;
    esac
  done
}

main $@
