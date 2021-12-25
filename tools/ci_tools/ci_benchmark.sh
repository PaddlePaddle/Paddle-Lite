#!/bin/bash
shopt -s expand_aliases
set -ex

NUM_CORES_FOR_COMPILE=${LITE_BUILD_THREADS:-8}
readonly EXE="benchmark_bin"

# Global variables
# Absolute path of Paddle-Lite source code.
SHELL_FOLDER=$(
  cd "$(dirname "$0")"
  pwd
)
WORKSPACE=${SHELL_FOLDER%tools/ci_tools*}
# ModelS URL
MODELS_URL="https://paddle-inference-dist.bj.bcebos.com/AI-Rank/mobile/MobileNetV1.tar.gz
            https://paddle-inference-dist.bj.bcebos.com/AI-Rank/mobile/MobileNetV2.tar.gz
            https://paddle-inference-dist.bj.bcebos.com/AI-Rank/mobile/MobileNetV3_large_x1_0.tar.gz
            https://paddle-inference-dist.bj.bcebos.com/AI-Rank/mobile/MobileNetV3_small_x1_0.tar.gz
            https://paddle-inference-dist.bj.bcebos.com/AI-Rank/mobile/ResNet50.tar.gz
            https://paddle-inference-dist.bj.bcebos.com/AI-Rank/mobile/ssdlite_mobilenet_v3_large.tar.gz
           "
# Download models everytime or not
FORCE_DOWNLOAD_MODELS="ON"
# Model zoo path on host
HOST_MODEL_ZOO_DIR=${WORKSPACE}/Models
# Config json file stored models, devices and performace data
CONFIG_PATH=${WORKSPACE}/tools/ci_tools/ci_benchmark_config.json
# The list of os for building(android,armlinux,linux,macos), such as "android"
OS_LIST="android"
# The list of arch abi for building(armv8,armv7,armv7hf), such as "armv8,armv7"
# for android devices, "armv8" for RK3399, "armv7hf" for Raspberry pi 3B
ARCH_LIST="armv8"
# The list of toolchains for building(gcc,clang), such as "clang"
# for android, "gcc" for armlinx
TOOLCHAIN_LIST="clang"
# Remote device type:
# 0: adb for android devices; 1: ssh for armlinux devices; 2: for local device)
REMOTE_DEVICE_TYPE=0
# The list of the device names for the real android devices, use commas to separate them, such as "bcd71650,8MY0220C22019318,A49BEMHY79"
# The list of the device infos for the real armlinux devices, its format is "dev0_ip_addr,dev0_port,dev0_usr_id,dev0_usr_pwd:dev1_ip_addr,dev0_port,dev1_usr_id,dev1_usr_pwd"
REMOTE_DEVICE_LIST="2GX0119401000796,0123456789ABCDEF"
# Work directory of the remote devices for running
REMOTE_DEVICE_WORK_DIR="/data/local/tmp/benchmark_ci_test/"
WARMUP=20
REPEATS=600

# Helper functions
source ${SHELL_FOLDER}/utils.sh

# if operating in mac env, we should expand the maximum file num
os_name=$(uname -s)
if [ ${os_name} == "Darwin" ]; then
  ulimit -n 1024
fi

# Build target function
function build_target() {
  local os=$1
  local arch=$2
  local toolchain=$3

  # Remove Compiling Cache
  rm -rf build.*

  # Compiling
  if [[ "$os" == "armlinux" ]]; then
    os="linux"
  fi
  echo "$PWD"
  cmd_line="./lite/tools/build_${os}.sh --arch=$arch --toolchain=$toolchain --with_benchmark=ON full_publish"
  ${cmd_line}

  # Checking results
  local exe_file=$(ls build.*/lite/api/${EXE})
  if [ ! -f $exe_file ]; then
    echo -e "$RED_COLOR $exe_file is not exist! $OFF_COLOR"
    echo -e "Compiling task failed on the following instruction:\n $cmd_line"
    exit 1
  fi
}

# Check benchmark reuslt
function check_benchmark_result() {
  local res_file=$1
  local config_path=$2
  local os=$3
  local arch=$4
  local toolchain=$5
  local remote_device_name=$6
  local model_name=$7
  local backend=$8

  if [[ "$config_path" == "" ]]; then
    echo -e "$YELOW_COLOR $config_path is not set! Skip result check! $OFF_COLOR"
    return 1
  fi

  local toolchain_in_config=`jq -r .toolchain $config_path`
  # Skip avg time check if toolchain not matched
  if [[ "$toolchain" != "$toolchain_in_config" ]]; then
    echo -e "$RED_COLOR Build with toolchain is $toolchain, while toolchain in $config_path is $toolchain_in_config. They are not matched! Skip avg time check! $OFF_COLOR"
    return 1
  fi
  local key_avg_time="avg"
  local avg_time=$(grep $key_avg_time $res_file | awk '{print $3}' | tail -1)
  local avg_time_baseline=`jq -r --arg v1 $model_name \
                                  --arg v2 $backend \
                                  --arg v3 $arch \
                                  --arg v4 $remote_device_name \
                                  '.model[] | select(.name == $v1) | .backends[] | select(.name == $v2) | .arch[] | select(.name == $v3) | .device_id[] | select(.name == $v4).avg_time_baseline' $config_path`
  local avg_time_thres_scale=`jq -r --arg v1 $model_name \
                                  --arg v2 $backend \
                                  --arg v3 $arch \
                                  --arg v4 $remote_device_name \
                                  '.model[] | select(.name == $v1) | .backends[] | select(.name == $v2) | .arch[] | select(.name == $v3) | .device_id[] | select(.name == $v4).avg_time_thres_scale' $config_path`

  local avg_time_thres=$(echo "${avg_time_baseline}*${avg_time_thres_scale}" | bc)
  local device_alias=`jq -r --arg v1 $model_name \
                            --arg v2 $backend \
                            --arg v3 $arch \
                            --arg v4 $remote_device_name \
                            '.model[] | select(.name == $v1) | .backends[] | select(.name == $v2) | .arch[] | select(.name == $v3) | .device_id[] | select(.name == $v4).alias' $config_path`
  if [ 1 -eq "$(echo "${avg_time} > ${avg_time_thres}" | bc)" ]; then
    echo -e "$RED_COLOR avg_time[${avg_time}] > avg_time_thres[${avg_time_thres}] on device[$device_alias] !\nThis PR may reduce performace. Reject this PR. $OFF_COLOR"
    exit 1
  else
    echo -e "$GREEN_COLOR avg_time[${avg_time}] <= avg_time_thres[${avg_time_thres}] on device[$device_alias] Passed. $OFF_COLOR"
    # TODO: update .json automatically(after this pr is merged)
    # sed -i "s/\${avg_time_baseline}\b/${avg_time}/" $config_path
  fi
  return 0
}

function run_on_remote_device() {
  local os=""
  local arch=""
  local toolchain=""
  local remote_device_name=""
  local remote_device_work_dir=""
  local remote_device_run=""
  local target_name=""
  local model_dir=""
  local config_path=""

  # Extract arguments from command line
  for i in "$@"; do
    case $i in
    --os=*)
      os="${i#*=}"
      shift
      ;;
    --arch=*)
      arch="${i#*=}"
      shift
      ;;
    --toolchain=*)
      toolchain="${i#*=}"
      shift
      ;;
    --remote_device_name=*)
      remote_device_name="${i#*=}"
      shift
      ;;
    --remote_device_work_dir=*)
      remote_device_work_dir="${i#*=}"
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
    --config_path=*)
      config_path="${i#*=}"
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
    echo -e "$RED_COLOR $target_name not found! $OFF_COLOR"
    exit 1
  fi
  $remote_device_run $remote_device_name shell "rm -f $remote_device_work_dir/$target_name"
  $remote_device_run $remote_device_name push "$target_path" "$remote_device_work_dir"

  # Get command line arguments
  local cmd_line=""
  local model_file=`$remote_device_run $remote_device_name shell "cd ${remote_device_work_dir}; ls ${model_dir}/*.pdmodel"`
  local param_file=`$remote_device_run $remote_device_name shell "cd ${remote_device_work_dir}; ls ${model_dir}/*.pdiparams"`
  local model_name=$(basename $model_dir)
  local input_shape=`jq -r --arg v $model_name '.model[] | select(.name == $v).input_shape' $config_path`
  local backends=""
  if [[ "$os" == "android" || "$os" == "armlinux" ]]; then
    backends=("arm" "opencl,arm")
  elif [ "$os" == "linux" ]; then
    backends=("x86")
  elif [ "$os" == "macos" ]; then
    backends=("x86" "opencl,x86")
  fi

  for backend in ${backends[@]}; do
    local res_file="result_${model_name}_${arch}_${toolchain}_${backend}_${remote_device_name}.txt"
    cmd_line="
              ./$target_name \
              --model_file=$model_file \
              --param_file=$param_file \
              --input_shape=$input_shape \
              --backend=$backend \
              --warmup=$WARMUP \
              --repeats=$REPEATS \
              --result_path=$res_file \
             "
    echo -e "$GREEN_COLOR model:${model_name} arch:${arch} toolchain:${toolchain} backend:${backend} device:${remote_device_name} $OFF_COLOR"
    echo "cmd_line start..."
    # Run the model on the remote device
    $remote_device_run $remote_device_name shell "cd $remote_device_work_dir; LD_LIBRARY_PATH=$LD_LIBRARY_PATH:.; rm -rf $res_file; $cmd_line; cd -"
    echo "cmd_line end"
    $remote_device_run $remote_device_name pull "${remote_device_work_dir}/${res_file}" .

    # Check benchmark result
    check_benchmark_result $res_file $config_path $os $arch $toolchain $remote_device_name $model_name $backend
  done
}

function build_and_test_on_remote_device() {
  local os_list=$1
  local arch_list=$2
  local toolchain_list=$3
  local build_target_func=$4
  local prepare_device_func=$5
  local host_model_zoo_dir=$6
  local force_download_models=$7
  local config_path=$8
  local remote_device_type=$9
  local remote_device_list=${10}
  local remote_device_work_dir=${11}
  local extra_arguments=${12}

  # 0. Set helper functions to access the remote devices
  local remote_device_pick=ssh_device_pick
  local remote_device_check=ssh_device_check
  local remote_device_run=ssh_device_run
  if [[ $remote_device_type -eq 0 ]]; then
    remote_device_pick=adb_device_pick
    remote_device_check=adb_device_check
    remote_device_run=adb_device_run
  fi

  # 1. Check remote devices are available or not
  local remote_device_names=$($remote_device_pick $remote_device_list)
  if [[ -z $remote_device_names ]]; then
      echo "No remote device available!"
      exit 1
  else
      echo "Found device(s) $remote_device_names."
  fi

  cd $PWD

  # 2. Download models to host machine
  prepare_models $host_model_zoo_dir $force_download_models

  # 3. Prepare device environment for running, such as device check and push models to remote device, only once for one device
  for remote_device_name in $remote_device_names; do
    $prepare_device_func $remote_device_name $remote_device_work_dir $remote_device_check $remote_device_run $host_model_zoo_dir
  done

  # 4. Run
  local oss=(${os_list//,/ })
  local archs=(${arch_list//,/ })
  local toolchains=(${toolchain_list//,/ })
  for os in ${oss[@]}; do
    for arch in ${archs[@]}; do
      for toolchain in ${toolchains[@]}; do
        # Build
        echo "Build with $os+$arch+$toolchain ..."
        $build_target_func $os $arch $toolchain
        # TODO: only tested on android currently
        if [[ "$os" != "android" ]]; then
          continue
        fi
        # Loop all remote devices
        for remote_device_name in $remote_device_names; do
          # Loop all models
          for model_dir in $(ls $host_model_zoo_dir); do
            # Run
            run_on_remote_device \
              --os=$os \
              --arch=$arch \
              --toolchain=$toolchain \
              --remote_device_name=$remote_device_name \
              --remote_device_work_dir=$remote_device_work_dir \
              --remote_device_run=$remote_device_run \
              --target_name=$EXE \
              --model_dir=$(basename ${host_model_zoo_dir})/${model_dir} \
              --config_path=$config_path
          done
        done
        cd - >/dev/null
      done
    done
  done
}

function android_build_and_test() {
  build_and_test_on_remote_device $OS_LIST $ARCH_LIST $TOOLCHAIN_LIST \
      build_target android_prepare_device \
      $HOST_MODEL_ZOO_DIR $FORCE_DOWNLOAD_MODELS \
      $CONFIG_PATH \
      $REMOTE_DEVICE_TYPE $REMOTE_DEVICE_LIST $REMOTE_DEVICE_WORK_DIR
}

function check_command_exist() {
  local cmd=$1
  which "$cmd" >/dev/null 2>&1
  if [ $? -ne 0 ]; then
    echo -e "$RED_COLOR $cmd is not found! $OFF_COLOR"
    exit 1
  fi
}

function main() {
  # Check requirements
  check_command_exist "jq"

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
    --host_model_zoo_dir=*)
      HOST_MODEL_ZOO_DIR="${i#*=}"
      shift
      ;;
    --force_download_models=*)
      FORCE_DOWNLOAD_MODELS="${i#*=}"
      shift
      ;;
    --config_path=*)
      CONFIG_PATH="${i#*=}"
      shift
      ;;
    --warmup=*)
      WARMUP="${i#*=}"
      shift
      ;;
    --repeats=*)
      REPEATS="${i#*=}"
      shift
      ;;
    android_build_and_test)
      android_build_and_test
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
