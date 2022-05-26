#!/bin/bash
set -ex

# Color
RED_COLOR='\E[1;31m'   # red
GREEN_COLOR='\E[1;32m' # green
YELOW_COLOR='\E[1;33m' # yellow
BLUE_COLOR='\E[1;34m'  # blue
PINK='\E[1;35m'        # pink
OFF_COLOR='\E[0m'      # off color

# URL that stores third-party tar.gz file to accelerate third-party lib installation
readonly THIRDPARTY_URL=https://paddlelite-data.bj.bcebos.com/third_party_libs/
readonly THIRDPARTY_TAR=third-party-91a9ab3.tar.gz

# wget wrapper for download file from Baidu bcebos
function wget_wrapper() {
  local url=$1
  local bak_http_proxy=""
  local bak_https_proxy=""
  if [ $http_proxy ]; then
    bak_http_proxy=$http_proxy
    bak_https_proxy=$https_proxy
    unset http_proxy
    unset https_proxy
  fi

  wget $url

  if [ $bak_http_proxy ]; then
    export http_proxy=$bak_http_proxy
    export https_proxy=$bak_https_proxy
  fi
}

function prepare_thirdparty() {
    local workspace=$1
    cd $workspace
    if [ ! -d $workspace/third-party -o -f $workspace/$THIRDPARTY_TAR ]; then
        rm -rf $workspace/third-party

        if [ ! -f $workspace/$THIRDPARTY_TAR ]; then
            wget_wrapper $THIRDPARTY_URL/$THIRDPARTY_TAR
        fi
        tar xzf $THIRDPARTY_TAR
    else
        git submodule update --init --recursive
    fi
    cd -
}

# Download models into user specific directory
function prepare_models {
  local model_zoo_dir=$1
  local force_download=$2

  if [[ "$force_download" == "OFF" && -d "$model_zoo_dir" ]]; then
    return 0
  fi

  rm -rf $model_zoo_dir && mkdir $model_zoo_dir && cd $model_zoo_dir
  # download compressed model recorded in $MODELS_URL
  for url in ${MODELS_URL[@]}; do
    wget_wrapper $url
  done

  compressed_models=$(ls)
  # decompress models
  for name in ${compressed_models[@]}; do
    if echo "$name" | grep -q -E '.tar.gz$'; then
      tar xf $name && rm -f $name
    elif echo "$name" | grep -q -E '.zip$'; then
      unzip $name && rm -f $name
    else
      echo "Error, only .zip or .tar.gz format files are supported!"
      exit 1
    fi
  done
  cd -
}

# Check one device validation
function adb_device_check() {
  local adb_device_name=$1
  if [[ -n "$adb_device_name" ]]; then
    for line in $(adb devices | grep -v "List" | grep device | awk '{print $1}'); do
      online_device_name=$(echo $line | awk '{print $1}')
      if [[ "$adb_device_name" == "$online_device_name" ]]; then
        return 0
      fi
    done
  fi
  return 1
}

# Pick one or more devices
function adb_device_pick() {
  local names=""
  local adb_device_list=$1
  local adb_device_names=(${adb_device_list//,/ })
  for adb_device_name in ${adb_device_names[@]}; do
    adb_device_check $adb_device_name
    if [[ $? -eq 0 ]]; then
      names="$names $adb_device_name"
    else
      return 1
    fi
  done
  echo $names
  return 0
}

function adb_device_run() {
  local adb_device_name=$1
  local adb_device_cmd=$2
  if [[ "$adb_device_cmd" == "shell" ]]; then
    adb -s $adb_device_name shell "$3"
  elif [[ "$adb_device_cmd" == "push" ]]; then
    local src_path=$3
    local dst_path=$4
    # adb push don't support '/*', so replace it with '/.'
    if [[ ${#src_path} -gt 2 ]]; then
      local src_suffix=${src_path: -2}
      if [[ "$src_suffix" == "/*" ]]; then
        src_path=${src_path:0:-2}/.
      fi
    fi
    adb -s $adb_device_name push "$src_path" "$dst_path"
  elif [[ "$adb_device_cmd" == "pull" ]]; then
    local src_path=$3
    local dst_path=$4
    # adb pull don't support '/*', so replace it with '/.'
    if [[ ${#src_path} -gt 2 ]]; then
      local src_suffix=${src_path: -2}
      if [[ "$src_suffix" == "/*" ]]; then
        src_path=${src_path:0:-2}/.
      fi
    fi
    adb -s $adb_device_name pull "$src_path" "$dst_path"
  elif [[ "$adb_device_cmd" == "root" ]]; then
    adb -s $adb_device_name root
  elif [[ "$adb_device_cmd" == "remount" ]]; then
    adb -s $adb_device_name remount
  else
    echo "Unknown command $adb_device_cmd"
    exit 1
  fi
}

function ssh_device_check() {
  local ssh_device_name=$1
  ssh_device_run $ssh_device_name test
}

function ssh_device_pick() {
  local ssh_device_list=$1
  local ssh_device_names=(${ssh_device_list//:/ })
  for ssh_device_name in ${ssh_device_names[@]}; do
    ssh_device_check $ssh_device_name
    if [[ $? -eq 0 ]]; then
      echo $ssh_device_name
      return 0
    fi
  done
  echo ""
  return 1
}

function ssh_device_run() {
  local ssh_device_name=$1
  local ssh_device_cmd=$2
  if [[ -z "$ssh_device_name" ]]; then
    echo "SSH device name is empty!"
    exit 1
  fi
  local ssh_device_items=(${ssh_device_name//,/ })
  if [[ ${#ssh_device_items[@]} -ne 4 ]]; then
    echo "SSH device name parse failed!"
    exit 1
  fi
  local ssh_device_ip_addr=${ssh_device_items[0]}
  local ssh_device_port=${ssh_device_items[1]}
  local ssh_device_usr_id=${ssh_device_items[2]}
  local ssh_device_usr_pwd=${ssh_device_items[3]}
  if [[ -z "$ssh_device_ip_addr" || -z "$ssh_device_usr_id" ]]; then
    echo "SSH device IP Address or User ID is empty!"
    exit 1
  fi
  if [[ "$ssh_device_cmd" == "shell" ]]; then
    sshpass -p $ssh_device_usr_pwd ssh -o ConnectTimeout=60 -o StrictHostKeyChecking=no -p $ssh_device_port $ssh_device_usr_id@$ssh_device_ip_addr "$3"
  elif [[ "$ssh_device_cmd" == "push" ]]; then
    sshpass -p $ssh_device_usr_pwd scp -r -o ConnectTimeout=60 -o StrictHostKeyChecking=no -P $ssh_device_port $3 $ssh_device_usr_id@$ssh_device_ip_addr:$4
  elif [[ "$ssh_device_cmd" == "pull" ]]; then
    sshpass -p $ssh_device_usr_pwd scp -r -o ConnectTimeout=60 -o StrictHostKeyChecking=no -P $ssh_device_port $ssh_device_usr_id@$ssh_device_ip_addr:$4 $3
  elif [[ "$ssh_device_cmd" == "test" ]]; then
    sshpass -p $ssh_device_usr_pwd ssh -o ConnectTimeout=60 -o StrictHostKeyChecking=no -p $ssh_device_port $ssh_device_usr_id@$ssh_device_ip_addr "exit 0" &>/dev/null
  else
    echo "Unknown command $ssh_device_cmd!"
    exit 1
  fi
}

function android_prepare_device() {
  local remote_device_name=$1
  local remote_device_work_dir=$2
  local remote_device_check=$3
  local remote_device_run=$4
  local model_dir=$5

  # Check device is available
  $remote_device_check $remote_device_name
  if [[ $? -ne 0 ]]; then
    echo "$remote_device_name not found!"
    exit 1
  fi

  # Create work dir on the remote device
  if [[ -z "$remote_device_work_dir" ]]; then
    echo "$remote_device_work_dir can't be empty!"
    exit 1
  fi
  if [[ "$remote_device_work_dir" == "/" ]]; then
    echo "$remote_device_work_dir can't be root dir!"
    exit 1
  fi

  # Create work dir & push model if necessary
  $remote_device_run $remote_device_name shell "rm -rf $remote_device_work_dir"
  $remote_device_run $remote_device_name shell "mkdir -p $remote_device_work_dir"
  if [[ -n "$model_dir" ]]; then
    $remote_device_run $remote_device_name push $model_dir $remote_device_work_dir
  fi
}

# Generate `__generated_code__.cc`, which is dependended by some targets in cmake.
# here we fake an empty file to make cmake works.
function prepare_workspace() {
    local root_dir=$1
    local build_dir=$2

    # 1. Prepare gen_code file
    local GEN_CODE_PATH_PREFIX=$build_dir/lite/gen_code
    mkdir -p ${GEN_CODE_PATH_PREFIX}
    touch ${GEN_CODE_PATH_PREFIX}/__generated_code__.cc

    # 2.Prepare debug tool
    local DEBUG_TOOL_PATH_PREFIX=$build_dir/lite/tools/debug
    mkdir -p ${DEBUG_TOOL_PATH_PREFIX}
    cp $root_dir/lite/tools/debug/analysis_tool.py ${DEBUG_TOOL_PATH_PREFIX}/

    prepare_thirdparty $root_dir
}

# Prepare source code of opencl lib
# here we bundle all cl files into a cc file to bundle all opencl kernels into a single lib
function prepare_opencl_source_code() {
    local root_dir=$1
    # in build directory
    # Prepare opencl_kernels_source.cc file
    GEN_CODE_PATH_OPENCL=$root_dir/lite/backends/opencl
    rm -f GEN_CODE_PATH_OPENCL/opencl_kernels_source.cc
    OPENCL_KERNELS_PATH=$root_dir/lite/backends/opencl/cl_kernel
    mkdir -p ${GEN_CODE_PATH_OPENCL}
    touch $GEN_CODE_PATH_OPENCL/opencl_kernels_source.cc
    python $root_dir/lite/tools/cmake_tools/gen_opencl_code.py $OPENCL_KERNELS_PATH $GEN_CODE_PATH_OPENCL/opencl_kernels_source.cc
}
