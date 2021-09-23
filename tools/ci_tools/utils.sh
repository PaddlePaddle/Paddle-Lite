#!/bin/bash
set -ex

# Download models into user specifical directory
function prepare_models {
  rm -rf $1 && mkdir $1 && cd $1
  # download compressed model recorded in $MODELS_URL
  for url in ${MODELS_URL[@]}; do
    wget $url
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
}

# Check one device validation
function adb_device_check() {
  local adb_device_name=$1
  if [[ -n "$adb_device_name" ]]; then
    for line in $(adb devices | grep -v "List" | awk '{print $1}'); do
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
  local adb_device_list=$1
  local adb_device_names=(${adb_device_list//,/ })
  for adb_device_name in ${adb_device_names[@]}; do
    adb_device_check $adb_device_name
    if [[ $? -eq 0 ]]; then
      echo $adb_device_name "is available."
    else
      echo $adb_device_name "is not available!"
      return 1
    fi
  done
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
  elif [[ "$adb_device_cmd" == "root" ]]; then
    adb -s $adb_device_name root
  elif [[ "$adb_device_cmd" == "remount" ]]; then
    adb -s $adb_device_name remount
  else
    echo "Unknown command $adb_device_cmd!"
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
  elif [[ "$ssh_device_cmd" == "test" ]]; then
    sshpass -p $ssh_device_usr_pwd ssh -o ConnectTimeout=60 -o StrictHostKeyChecking=no -p $ssh_device_port $ssh_device_usr_id@$ssh_device_ip_addr "exit 0" &>/dev/null
  else
    echo "Unknown command $ssh_device_cmd!"
    exit 1
  fi
}

# Android
function android_prepare_device() {
  local os=$1
  local arch=$2
  local toolchain=$3
  local remote_device_name=$4
  local remote_device_work_dir=$5
  local remote_device_check=$6
  local remote_device_run=$7
  local model_dir=$8

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
  $remote_device_run $remote_device_name shell "rm -rf $remote_device_work_dir"
  $remote_device_run $remote_device_name shell "mkdir -p $remote_device_work_dir"
  $remote_device_run $remote_device_name push $model_dir $remote_device_work_dir
}
