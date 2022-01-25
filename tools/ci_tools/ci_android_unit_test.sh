#!/bin/bash
set +x
set -e

# Absolute path of Paddle-Lite source code.
SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
WORKSPACE=${SHELL_FOLDER%tools/ci_tools*}
TESTS_FILE="./lite_tests.txt"
NUM_PROC=4

# controls whether to include FP16 kernels, default is OFF
BUILD_ARM82_FP16=OFF

skip_list=("test_model_parser" "test_mobilenetv1" "test_mobilenetv2" \
            "test_resnet50" "test_inceptionv4" "test_light_api" "test_apis" \
            "test_paddle_api" "test_cxx_api" "test_gen_code" \
            "test_mobilenetv1_int8" "test_subgraph_pass" \
            "test_transformer_with_mask_fp32_arm" "test_mobilenet_v1_int8_per_layer_arm" \
            "test_mobilenetv1_int16" "test_mobilenetv1_opt_quant" \
            "test_fast_rcnn" "test_inception_v4_fp32_arm" "test_mobilenet_v1_fp32_arm" \
            "test_mobilenet_v2_fp32_arm" "test_mobilenet_v3_small_x1_0_fp32_arm" \
            "test_mobilenet_v3_large_x1_0_fp32_arm" "test_resnet50_fp32_arm" \
            "test_squeezenet_fp32_arm" "test_mobilenet_v1_int8_arm" \
            "test_mobilenet_v2_int8_arm" "test_resnet50_int8_arm" \
            "test_mobilenet_v1_int8_dygraph_arm" "test_ocr_lstm_int8_arm" \
            "get_conv_latency" "get_batchnorm_latency" "get_pooling_latency" \
            "get_fc_latency" "get_activation_latency" "test_generated_code" \
            "test_lac_crf_fp32_arm" "test_nlp_lstm_int8_arm" \
            "test_transformer_nlp2_fp32_arm" "test_lac_crf_fp32_int16_arm")

# if operating in mac env, we should expand the maximum file num
os_name=`uname -s`
if [ ${os_name} == "Darwin" ]; then
   ulimit -n 10240
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

####################################################################################################

# test the inference high level api
function test_arm_api {
    local adb_device=$1
    local adb_work_dir=$2
    local test_name="test_paddle_api"

    make $test_name -j$NUM_PROC

    local model_path=$(find . -name "lite_naive_model")
    local testpath=$(find ./lite -name ${test_name})


    adb -s $adb_device push $testpath /data/local/tmp/$adb_work_dir
    adb -s $adb_device shell chmod +x "/data/local/tmp/$adb_work_dir/$test_name"
    adb -s $adb_device push $model_path /data/local/tmp/$adb_work_dir
    adb -s $adb_device shell "cd /data/local/tmp/$adb_work_dir && ./$test_name --model_dir lite_naive_model"
}

# 2 function of compiling
# here we compile android lib and unit test.
function build_android {
  os=$1
  arch=$2
  toolchain=$3

  build_directory=$WORKSPACE/ci.android.$arch.$toolchain
  rm -rf $build_directory && mkdir -p $build_directory


  git submodule update --init --recursive
  prepare_workspace $WORKSPACE $build_directory

  cd $build_directory
  cmake .. \
      -DWITH_GPU=OFF \
      -DWITH_MKL=OFF \
      -DLITE_WITH_CUDA=OFF \
      -DLITE_WITH_X86=OFF \
      -DLITE_WITH_ARM=ON \
      -DWITH_ARM_DOTPROD=ON   \
      -DWITH_TESTING=ON \
      -DLITE_BUILD_EXTRA=ON \
      -DLITE_WITH_TRAIN=ON \
      -DLITE_WITH_ARM82_FP16=$BUILD_ARM82_FP16 \
      -DARM_TARGET_OS=$os -DARM_TARGET_ARCH_ABI=$arch -DARM_TARGET_LANG=$toolchain

  make lite_compile_deps -j$NUM_PROC
  cd - > /dev/null
}

function test_arm_unit_test {
  unit_test=$1
  adb_devices=$2
  adb_work_dir=$3
  unit_test_path=$(find ./lite -name $unit_test)
  adb -s $adb_devices push $unit_test_path /data/local/tmp/$adb_work_dir
  adb -s $adb_devices shell "cd /data/local/tmp/$adb_work_dir && ./$unit_test"
}

function build_test_android {
  arch=$1
  toolchain=$2
  adb_device=$3
  adb_workdir=$4
  is_fp16=$5

  build_android android $arch $toolchain

  adb_devices=($(adb devices |grep -v devices |grep device | awk -F " " '{print $1}'))

  build_directory=$WORKSPACE/ci.android.$arch.$toolchain
  cd $build_directory

  adb -s ${adb_devices[0]} shell "cd /data/local/tmp && rm -rf $adb_workdir && mkdir $adb_workdir"
  test_arm_api ${adb_devices[0]} $adb_work_dir

  adb -s ${adb_devices[0]} shell "cd /data/local/tmp && rm -rf $adb_workdir && mkdir $adb_workdir"
  if [ $arch == "armv7" ] && [ $toolchain == "clang" ] ; then
     return
  else
      for _test in $(cat $TESTS_FILE); do
          local to_skip=0
          for skip_name in ${skip_list[@]}; do
              if [ $skip_name == $_test ]; then
                 echo "to skip " $skip_name
                 to_skip=1
              fi
          done
          if [ $is_fp16 == "enable_fp16" ] && [ $arch == "armv7" ]; then
             # v7 fp16 this case supporting
             for $_test in {"conv_fp16_compute_test", "gemm_fp16_compute_test", "fc_fp16_compute_test", "pool_fp16_compute_test"}; do
                 echo "to skip " $_test
                 to_skip=1
             done
          fi

          if [[ $to_skip -eq 0 ]]; then
             test_arm_unit_test $_test ${adb_devices[0]} $adb_workdir
          fi
    done
  fi
  adb -s ${adb_devices[0]} shell "cd /data/local/tmp && rm -rf $adb_workdir"
}

# print help infomation
if [ $# -lt 1 ] ; then
    echo "Usage Explaination:"
    echo " (1) tools/ci_tools/ci_android_unit_test.sh adb_index adb_workname : compile and perform FP32 android unit tests. "
    echo "         eg. tools/ci_tools/ci_android_unit_test.sh 0 adb_work01 "
    echo " (2) tools/ci_tools/ci_android_unit_test.sh adb_index adb_workname  enable_fp16: compile and perform FP16 android unit tests. "
    echo "         eg. tools/ci_tools/ci_android_unit_test.sh 0 adb_work02 enable_fp16 "
    exit 0
fi

if [ $# -eq 3 ] && [ $3 == "enable_fp16" ] ; then
    BUILD_ARM82_FP16=ON
    build_test_android armv8 clang $1 $2 $3
    build_test_android armv7 clang $1 $2 $3
else
    # $1 adb_device index. eg. 1
    # $2 workspace name on adb.  eg. work_tmp1
    build_test_android armv7 gcc $1 $2 ""
    # only test build
    build_test_android armv7 clang $1 $2 ""
    build_test_android armv8 clang $1 $2 ""
fi
