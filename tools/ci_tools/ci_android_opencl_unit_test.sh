#!/bin/bash
set +x
set -e

# Absolute path of Paddle-Lite source code.
SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
WORKSPACE=${SHELL_FOLDER%tools/ci_tools*}
TESTS_FILE="./lite_tests.txt"
NUM_PROC=4

skip_list=("test_model_parser" "test_light_api" "test_apis" \
            "test_paddle_api" "test_cxx_api" "test_gen_code" \
            "test_mobilenetv1_int8" "test_subgraph_pass" \
            "test_grid_sampler_image_opencl" "test_lrn_image_opencl" \
            "test_pad2d_image_opencl" "test_transformer_with_mask_fp32_arm" \
            "test_mobilenetv1_int16" "test_mobilenetv1_opt_quant" \
            "test_fast_rcnn" "test_inception_v4_fp32_arm" "test_mobilenet_v1_fp32_arm" \
            "test_mobilenet_v2_fp32_arm" "test_mobilenet_v3_small_x1_0_fp32_arm" \
            "test_mobilenet_v3_large_x1_0_fp32_arm" "test_resnet50_fp32_arm" \
            "test_squeezenet_fp32_arm" "test_mobilenet_v1_int8_arm" \
            "test_mobilenet_v2_int8_arm" "test_resnet50_int8_arm" \
            "test_mobilenet_v1_int8_dygraph_arm" "test_ocr_lstm_int8_arm" \
            "get_conv_latency" "get_batchnorm_latency" "get_pooling_latency" \
            "get_fc_latency" "get_activation_latency" \
            "test_lac_crf_fp32_arm" "test_nlp_lstm_int8_arm" \
            "test_transformer_nlp2_fp32_arm")

# if operating in mac env, we should expand the maximum file num
os_name=`uname -s`
if [ ${os_name} == "Darwin" ]; then
   ulimit -n 4096 
fi

####################################################################################################
# 1. functions of prepare workspace before compiling
####################################################################################################

# 1.1 generate `__generated_code__.cc`, which is dependended by some targets in cmake.
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


# 1.2 prepare source code of opencl lib
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


# 2 function of opencl compiling
# here we compile android lib and unit test.
function build_opencl {
  os=$1
  arch=$2
  toolchain=$3

  build_directory=$WORKSPACE/ci.android.opencl.$arch.$toolchain
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

  make lite_compile_deps -j$NUM_PROC
  cd - > /dev/null
}

function test_arm_android {
  unit_test=$1
  adb_devices=$2
  adb_work_dir=$3
  unit_test_path=$(find ./lite -name $unit_test)
  adb -s $adb_devices push $unit_test_path /data/local/tmp/$adb_work_dir
  adb -s $adb_devices shell "cd /data/local/tmp/$adb_work_dir && ./$unit_test"
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

  build_directory=$WORKSPACE/ci.android.opencl.$arch.$toolchain
  cd $build_directory


  adb -s ${adb_devices[0]} shell "cd /data/local/tmp && rm -rf $adb_workdir && mkdir $adb_workdir"
  for _test in $(cat $TESTS_FILE); do
      local to_skip=0
      for skip_name in ${skip_list[@]}; do
          if [ $skip_name = $_test ]; then
              echo "to skip " $skip_name
              to_skip=1
          fi
      done

      # tell if this test is marked with `opencl`
      if [[ $_test == *$opencl_test_mark* ]] && [[ $to_skip -eq 0 ]]; then
          test_arm_android $_test ${adb_devices[0]} $adb_workdir
      fi
  done
  adb -s ${adb_devices[0]} shell "cd /data/local/tmp && rm -rf $adb_workdir"
}

# $1 adb_device index. eg. 1
# $2 workspace name on adb.  eg. work_tmp1
build_test_android_opencl armv7 gcc $1 $2
build_test_android_opencl armv8 gcc $1 $2
