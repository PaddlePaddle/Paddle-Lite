#!/bin/bash
set +x
set -e

# Absolute path of Paddle-Lite source code.
SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
WORKSPACE=${SHELL_FOLDER%tools/ci_tools*}
TESTS_FILE="./lite_tests.txt"

####################################################################################################
# 3. functions of prepare workspace before compiling
####################################################################################################

# 3.1 generate `__generated_code__.cc`, which is dependended by some targets in cmake.
# here we fake an empty file to make cmake works.
function prepare_workspace {
    local root_dir=$1
    local build_dir=$2
    # 1. Prepare gen_code file
    GEN_CODE_PATH_PREFIX=$build_dir/lite/gen_code
    mkdir -p ${GEN_CODE_PATH_PREFIX}
    touch ${GEN_CODE_PATH_PREFIX}/__generated_code__.cc
    # 2.Prepare debug tool
    DEBUG_TOOL_PATH_PREFIX=$build_dir/lite/tools/debug
    mkdir -p ${DEBUG_TOOL_PATH_PREFIX}
    cp $root_dir/lite/tools/debug/analysis_tool.py ${DEBUG_TOOL_PATH_PREFIX}/
}


# 3.2 prepare source code of opencl lib
# here we bundle all cl files into a cc file to bundle all opencl kernels into a single lib
function prepare_opencl_source_code {
    local root_dir=$1
    local build_dir=$2
    # in build directory
    # Prepare opencl_kernels_source.cc file
    GEN_CODE_PATH_OPENCL=$root_dir/lite/backends/opencl
    rm -f GEN_CODE_PATH_OPENCL/opencl_kernels_source.cc
    OPENCL_KERNELS_PATH=$root_dir/lite/backends/opencl/cl_kernel
    mkdir -p ${GEN_CODE_PATH_OPENCL}
    touch $GEN_CODE_PATH_OPENCL/opencl_kernels_source.cc
    python $root_dir/lite/tools/cmake_tools/gen_opencl_code.py $OPENCL_KERNELS_PATH $GEN_CODE_PATH_OPENCL/opencl_kernels_source.cc
}

####################################################################################################


# 4.2 function of full_publish compiling
# here we compile both light_api lib and full_api lib
function build_opencl {
  os=$1
  arch=$2
  toolchain=$3

  build_directory=$WORKSPACE/ci.android.opencl.$ARCH.$TOOLCHAIN
  rm -rf $build_directory && mkdir -p $build_directory


  git submodule update --init --recursive
  prepare_workspace $WORKSPACE $build_directory
  prepare_opencl_source_code $WORKSPACE $build_dir

  cd $build_directory
  cmake .. \
      -DLITE_WITH_OPENCL=ON \
      -DWITH_GPU=OFF \
      -DWITH_MKL=OFF \
      -DWITH_LITE=ON \
      -DLITE_WITH_CUDA=OFF \
      -DLITE_WITH_X86=OFF \
      -DLITE_WITH_ARM=ON \
      -DWITH_ARM_DOTPROD=ON   \
      -DLITE_WITH_LIGHT_WEIGHT_FRAMEWORK=ON \
      -DWITH_TESTING=ON \
      -DLITE_BUILD_EXTRA=ON \
      -DLITE_WITH_LOG=ON \
      -DLITE_WITH_CV=OFF \
      -DARM_TARGET_OS=$os -DARM_TARGET_ARCH_ABI=$arch -DARM_TARGET_LANG=$toolchain

  make opencl_clhpp -j$NUM_PROC
  make lite_compile_deps -j$NUM_PROC
  cd - > /dev/null
}
# test_arm_android <some_test_name> <adb_port_number>
function test_arm_android {
    local test_name=$1
    local device=$2
    local adb_work_dir=$3

    if [[ "${test_name}x" == "x" ]]; then
        echo "test_name can not be empty"
        exit 1
    fi
    if [[ "${device}x" == "x" ]]; then
        echo "Port can not be empty"
        exit 1
    fi

    echo "test name: ${test_name}"
    local testpath=$(find ./lite -name ${test_name})

    adb -s ${device} push ${testpath} ${adb_work_dir}
    adb -s ${device} shell "cd ${adb_work_dir} && ./${test_name}"
}
function build_test_android_opencl {
  arch=$1
  toolchain=$2
  adb_device=$3
  adb_workdir=$4

  build_opencl android $arch $toolchain

  # opencl test should be marked with `opencl`
  opencl_test_mark="opencl"

  adb_devices=($(adb devices |grep -v devices |grep device | awk -F " " '{print $1}'))

  for _test in $(cat $TESTS_FILE); do
      # tell if this test is marked with `opencl`
      if [[ $_test == *$opencl_test_mark* ]]; then
          test_arm_android $_test ${adb_devices[0]} $adb_workdir
      fi
  done
}

build_test_android_opencl armv7 gcc 0 areyouok
