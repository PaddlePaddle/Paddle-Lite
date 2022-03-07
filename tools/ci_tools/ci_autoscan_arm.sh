#!/bin/bash
# Start the CI task of unittest for op and pass.
set -x
set -e

#####################################################################################################
# 1. global variables, you can change them according to your requirements
#####################################################################################################
# Python version
PYTHON_VERSION=3.9
# Absolute path of Paddle-Lite source code.
SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
WORKSPACE=${SHELL_FOLDER%tools/ci_tools*}
# Common options, use commas to separate them, such as "ARM,OpenCL,Metal" or "ARM,OpenCL" or "ARM,Metal".
TARGET_LIST="ARM,OpenCL,Metal"
# Skip op or pass, use | to separate them, such as "expand_op" or "expand_op|abc_pass", etc.
SKIP_LIST="abc_op|abc_pass"
# Models URL
MODELS_URL="https://paddle-inference-dist.bj.bcebos.com/AI-Rank/mobile/MobileNetV1.tar.gz"

# Helper functions
source ${SHELL_FOLDER}/utils.sh

####################################################################################################
# Functions of operate unit test
# Arguments:
#   target_name: can be ARM or OpenCL or Metal
# Globals:
#   WORKSPACE
####################################################################################################
function auto_scan_test {
  local target_name=$1

  cd $WORKSPACE/lite/tests/unittest_py/rpc_service
  sh start_rpc_server.sh

  cd $WORKSPACE/lite/tests/unittest_py/op/
  unittests=$(ls | egrep -v $SKIP_LIST)
  for test in ${unittests[@]}; do
    if [[ "$test" =~ py$ ]]; then
      python3.8 $test --target=$target_name
    fi
  done

  cd $WORKSPACE/lite/tests/unittest_py/pass/
  unittests=$(ls | egrep -v $SKIP_LIST)
  for test in ${unittests[@]}; do
    if [[ "$test" =~ py$ ]]; then
      python3.8 $test --target=$target_name
    fi
  done

  cd $WORKSPACE/lite/tests/unittest_py/model_test/
  python3.8 run_model_test.py --target=$target_name
}

####################################################################################################
# Functions of compiling test.
# Arguments:
#   --target_list: can be ARM,OpenCL,Metal or ARM,OpenCL or ARM,Metal
# Globals:
#   WORKSPACE, PYTHON_VERSION
####################################################################################################
function compile_publish_inference_lib {
  local target_list=""
  # Extract arguments from command line
  for i in "$@"; do
    case $i in
      --target_list=*)
        target_list="${i#*=}"
        shift
        ;;
      *)
        shift
        ;;
    esac
  done

  local targets=(${target_list//,/ })
  local build_opencl=OFF
  local build_metal=OFF
  for target in ${targets[@]}; do
    if [[ "$target" == "OpenCL" ]]; then
      build_opencl=ON
    elif [[ "$target" == "Metal" ]]; then
      build_metal=ON
    fi
  done

  cd $WORKSPACE

  # Remove Compiling Cache
  rm -rf build.macos.*

  # Step1. Compiling python installer on mac M1
  local cmd_line="./lite/tools/build_macos.sh --with_python=ON --with_opencl=$build_opencl --with_metal=$build_metal --with_arm82_fp16=ON --python_version=$PYTHON_VERSION arm64"
  $cmd_line

  # Step2. Checking results: cplus and python inference lib
  local whl_path=$(find ./build.macos.armmacos.armv8.* -name *whl)
  if [[ -z "$whl_path" ]]; then
    # Error message.
    echo "**************************************************************************************"
    echo -e "$whl_path not found!"
    echo -e "Compiling task failed on the following instruction:\n $cmd_line"
    echo "**************************************************************************************"
    exit 1
  fi

  # Step3. Install whl and its depends
  python$PYTHON_VERSION -m pip install --force-reinstall $whl_path
  python3.8 -m pip install -r ./lite/tests/unittest_py/requirements.txt
}

function run_test() {
  local target_list=$1
  local targets=(${target_list//,/ })
  rm -rf $(find $WORKSPACE/lite/tests/unittest_py/ -name statics_data)

  for target in ${targets[@]}; do
    auto_scan_test $target
  done
}

function get_summary() {
  cd $WORKSPACE/lite/tests/unittest_py/op/
  python3.8 ../global_var_model.py
  cd $WORKSPACE/lite/tests/unittest_py/pass/
  python3.8 ../global_var_model.py
}

function check_classification_result() {
  local target=$1
  local log_file=$2
  local result_class_name="Egyptian cat"

  local ret=$(grep "$result_class_name" $log_file)
  if [[ -z "$ret" ]]; then
    echo "Wrong result on $target. exit!"
    exit 1
  fi
}

function run_python_demo() {
  local target_list=$1
  local targets=(${target_list//,/ })

  # Download model
  local download_dir="${WORKSPACE}/Models/"
  local force_download="ON"
  prepare_models $download_dir $force_download
  local model_dir=${download_dir}/$(ls $download_dir)

  # Requirements
  python$PYTHON_VERSION -m pip install opencv-python

  # Run demo & check result
  cd $WORKSPACE/lite/demo/python/
  local log_file="log"
  for target in ${targets[@]}; do
    # mobilenetv1_full_api
    python$PYTHON_VERSION mobilenetv1_full_api.py \
        --model_file ${model_dir}/inference.pdmodel \
        --param_file ${model_dir}/inference.pdiparams \
        --input_shape 1 3 224 224 \
        --label_path ./labels.txt \
        --image_path ./tabby_cat.jpg \
        --backend $target 2>&1 | tee $log_file
    check_classification_result $target $log_file

    # mobilenetv1_light_api
    python$PYTHON_VERSION mobilenetv1_light_api.py \
        --model_dir "opt_${target}.nb" \
        --input_shape 1 3 224 224 \
        --label_path ./labels.txt \
        --image_path ./tabby_cat.jpg \
        --backend $target 2>&1 | tee $log_file
    check_classification_result $target $log_file
  done
}

function pipeline() {
  # Compile
  compile_publish_inference_lib --target_list=$1

  # Run unittests
  run_test $1
}

function main() {
  # Parse command line.
  for i in "$@"; do
    case $i in
      --target_list=*)
        TARGET_LIST="${i#*=}"
        shift
        ;;
      --skip_list=*)
        SKIP_LIST="${i#*=}"
        shift
        ;;
      *)
        echo "Unknown option, exit"
        exit 1
        ;;
    esac
  done

  # Run op/pass unittests
  pipeline $TARGET_LIST
  get_summary

  # Run python demo
  run_python_demo $TARGET_LIST

  # Uninstall paddlelite
  python$PYTHON_VERSION -m pip uninstall -y paddlelite

  echo "Success for targets:" $TARGET_LIST
}

main $@
