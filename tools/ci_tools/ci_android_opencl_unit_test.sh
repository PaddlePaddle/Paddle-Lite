#!/bin/bash
shopt -s expand_aliases
set -ex

NUM_CORES_FOR_COMPILE=${LITE_BUILD_THREADS:-8}

# Global variables
# Absolute path of Paddle-Lite source code.
SHELL_FOLDER=$(
  cd "$(dirname "$0")"
  pwd
)
WORKSPACE=${SHELL_FOLDER%tools/ci_tools*}

readonly OPENCL_UTEST_MASK="opencl"
readonly TESTS_FILE="./lite_tests.txt"
# The list of arch abi for building(armv8,armv7,armv7hf), such as "armv8,armv7"
# for android devices, "armv8" for RK3399, "armv7hf" for Raspberry pi 3B
ARCH_LIST="armv8"
# The list of toolchains for building(gcc,clang), such as "clang"
# for android, "gcc" for armlinx
TOOLCHAIN_LIST="clang"
# The list of the device names for the real android devices, use commas to separate them, such as "bcd71650,8MY0220C22019318,A49BEMHY79"
REMOTE_DEVICE_LIST="bcd71650"
# Work directory of the remote devices for running
REMOTE_DEVICE_WORK_DIR="/data/local/tmp/ci_opencl_utest/"
# Skip utests whose name has specific keys
SKIP_UTEST_KEYS="_arm,_x86,lrn,grid,conv"
# The Logging level of GLOG for unit tests
UTEST_LOG_LEVEL=5

# Helper functions
source ${SHELL_FOLDER}/utils.sh

# if operating in mac env, we should expand the maximum file num
os_name=$(uname -s)
if [ ${os_name} == "Darwin" ]; then
  ulimit -n 1024
fi

# Build target function
# here we compile android lib and unit test for opencl.
function build_target {
  arch=$1
  toolchain=$2

  build_directory=$WORKSPACE/ci.android.opencl.$arch.$toolchain
  rm -rf $build_directory && mkdir -p $build_directory

  prepare_workspace $WORKSPACE $build_directory
  prepare_opencl_source_code $WORKSPACE

  cd $build_directory
  cmake .. \
      -DLITE_WITH_OPENCL=ON \
      -DWITH_GPU=OFF \
      -DWITH_MKL=OFF \
      -DLITE_WITH_CUDA=OFF \
      -DLITE_WITH_X86=OFF \
      -DLITE_WITH_ARM=ON \
      -DWITH_ARM_DOTPROD=ON   \
      -DWITH_TESTING=ON \
      -DLITE_BUILD_EXTRA=ON \
      -DLITE_WITH_LOG=ON \
      -DLITE_WITH_CV=OFF \
      -DARM_TARGET_OS=android -DARM_TARGET_ARCH_ABI=$arch -DARM_TARGET_LANG=$toolchain

  make lite_compile_deps -j$NUM_CORES_FOR_COMPILE
  cd -
}

function run_on_remote_device() {
  local remote_device_name=""
  local remote_device_work_dir=""
  local target_name=""
  local model_dir=""
  local data_dir=""

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
    --target_name=*)
      target_name="${i#*=}"
      shift
      ;;
    --model_dir=*)
      model_dir="${i#*=}"
      shift
      ;;
    --data_dir=*)
      data_dir="${i#*=}"
      shift
      ;;
    *)
      shift
      ;;
    esac
  done

  # Copy the executable to the remote device
  local target_path=$(find $WORKSPACE/ci.android.opencl.* -name $target_name)
  if [[ -z "$target_path" ]]; then
    echo -e "$target_name not found!"
    exit 1
  fi
  adb -s $remote_device_name shell "rm -f $remote_device_work_dir/$target_name"
  adb -s $remote_device_name push "$target_path" "$remote_device_work_dir"

  local command_line="./$target_name"
  # Copy the model files to the remote device
  if [[ -n "$model_dir" ]]; then
    local model_name=$(basename $model_dir)
    adb -s $remote_device_name shell "rm -rf $remote_device_work_dir/$model_name"
    adb -s $remote_device_name push "$model_dir" "$remote_device_work_dir"
    command_line="$command_line --model_dir ./$model_name"
  fi

  # Copy the test data files to the remote device
  if [[ -n "$data_dir" ]]; then
    local data_name=$(basename $data_dir)
    adb -s $remote_device_name shell "rm -rf $remote_device_work_dir/$data_name"
    adb -s $remote_device_name push "$data_dir" "$remote_device_work_dir"
    command_line="$command_line --data_dir ./$data_name"
  fi

  # Run
  adb -s $remote_device_name shell "cd $remote_device_work_dir; export GLOG_v=$UTEST_LOG_LEVEL; $command_line"
}

function build_and_test_on_remote_device() {
  local arch_list=$1
  local toolchain_list=$2
  local build_target_func=$3
  local prepare_device_func=$4
  local remote_device_list=$5
  local remote_device_work_dir=$6

  echo "remote_device_work_dir: " $remote_device_work_dir

  # 1. Check remote devices are available or not
  local remote_device_names=$($adb_device_pick $remote_device_list)
  if [[ -z $remote_device_names ]]; then
    echo "No remote device available! Try pick one remote device..."
    local adb_devices=($(adb devices |grep -v devices |grep device | awk -F " " '{print $1}'))
    remote_device_names=${adb_devices[0]}
    if [ -n $device_serial ]; then
      echo "Found one device $remote_device_names."
    else
      echo "No available device!"
      exit 1
    fi
  else
    echo "Found device(s) $remote_device_names."
  fi

  cd $PWD

  # 2. Prepare device environment for running, such as device check, only once for one device
  for remote_device_name in $remote_device_names; do
    $prepare_device_func $remote_device_name $remote_device_work_dir adb_device_check adb_device_run
  done

  # 3. Build & Run
  local archs=(${arch_list//,/ })
  local toolchains=(${toolchain_list//,/ })
  for arch in ${archs[@]}; do
    for toolchain in ${toolchains[@]}; do
      # Build
      echo "Build with $arch+$toolchain ..."
      $build_target_func $arch $toolchain
      cd ${PWD}/ci.android.opencl*

      # Loop all test_name
      for test_name in $(cat $TESTS_FILE); do
        # Skip some utests
        local skip_keys=(${SKIP_UTEST_KEYS//,/ })
        local to_skip=0
        for skip_key in ${skip_keys[@]}; do
          if [[ $test_name == *${skip_key}* ]]; then
            echo "Skip utest " $test_name
            to_skip=1
            break;
          fi
        done

        # Extract the arguments from ctest command line
        test_cmds=$(ctest -V -N -R ^$test_name$)
        reg_expr=".*Test command:.*\/$test_name \(.*\) Test #[0-9]*: $test_name.*"
        test_args=$(echo $test_cmds | sed -n "/$reg_expr/p")
        echo "test_cmds: " $test_cmds
        echo "test_args 1 : " $test_args
        if [[ -n "$test_args" ]]; then
          # Matched, extract and remove the quotes
          test_args=$(echo $test_cmds | sed "s/$reg_expr/\1/g")
          test_args=$(echo $test_args | sed "s/\"//g")
        fi

        echo "test_args 2 : " $test_args

        # Tell if this test is marked with `opencl`
        if [[ $test_name == *$OPENCL_UTEST_MASK* ]] && [[ $to_skip -eq 0 ]]; then
          # Loop all remote devices
          for remote_device_name in $remote_device_names; do
            # Run
            run_on_remote_device \
                --remote_device_name=$remote_device_name \
                --remote_device_work_dir=$remote_device_work_dir \
                --target_name=$test_name \
                $test_args
          done
        fi
      done
    done
  done
  cd - >/dev/null
}

function android_build_and_test() {
  build_and_test_on_remote_device $ARCH_LIST $TOOLCHAIN_LIST \
      build_target android_prepare_device \
      $REMOTE_DEVICE_LIST $REMOTE_DEVICE_WORK_DIR
}

function main() {
  # Parse command line.
  for i in "$@"; do
    case $i in
    --arch_list=*)
      ARCH_LIST="${i#*=}"
      shift
      ;;
    --toolchain_list=*)
      TOOLCHAIN_LIST="${i#*=}"
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

start_time=`date +%Y%m%d-%H:%M:%S`
start_time_s=`date +%s`
main android_build_and_test
end_time=`date +%Y%m%d-%H:%M:%S`
end_time_s=`date +%s`
cost_ime_m=`echo "($end_time_s - $start_time_s) / 60" | bc`
echo "Start time: $start_time ---> End time: $end_time" "  This CI costs: $cost_ime_m minutes."
