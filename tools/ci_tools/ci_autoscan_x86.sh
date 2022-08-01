#!/bin/bash
# Start the CI task of unittest for op and pass.
set -x
set -e

#####################################################################################################
# 1. global variables, you can change them according to your requirements
#####################################################################################################
# Python version
PYTHON_VERSION=3.7
# Absolute path of Paddle-Lite source code.
SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
WORKSPACE=${SHELL_FOLDER%tools/ci_tools*}
# Common options
BUILD_EXTRA=ON
# Skip op or pass, use | to separate them, such as "expand_op" or "expand_op|abc_pass", etc.
SKIP_LIST="abc_op|abc_pass"
# Models URL
MODELS_URL="https://paddle-inference-dist.bj.bcebos.com/AI-Rank/mobile/MobileNetV1.tar.gz"

# Helper functions
source ${SHELL_FOLDER}/utils.sh

####################################################################################################
# Functions of operate unit test
# Arguments:
#   target_name: can be Host or X86
# Globals:
#   WORKSPACE
####################################################################################################
function auto_scan_test {
  local target_name=$1

  cd $WORKSPACE/lite/tests/unittest_py/op/
  unittests=$(ls | egrep -v $SKIP_LIST)
  for test in ${unittests[@]}; do
    if [[ "$test" =~ py$ ]]; then
      python$PYTHON_VERSION $test --target=$target_name
    fi
  done

  cd $WORKSPACE/lite/tests/unittest_py/pass/
  unittests=$(ls | egrep -v $SKIP_LIST)
  for test in ${unittests[@]}; do
    if [[ "$test" =~ py$ ]]; then
      python$PYTHON_VERSION $test --target=$target_name
    fi
  done

  cd $WORKSPACE/lite/tests/unittest_py/model_test/
  python3.7 run_model_test.py --target=$target_name
}

####################################################################################################
# Functions of compiling test.
# Globals:
#   WORKSPACE, PYTHON_VERSION
####################################################################################################
function compile_publish_inference_lib {
  cd $WORKSPACE

  # Remove Compiling Cache
  rm -rf build.lite.linux.x86.*

  # Step1. Compiling python installer
  local cmd_line="./lite/tools/build_linux.sh --with_python=ON --python_version=$PYTHON_VERSION --with_extra=$BUILD_EXTRA --arch=x86"
  $cmd_line

  # Step2. Checking results: cplus and python inference lib
  local whl_path=$(find ./build.lite.linux.x86.* -name *whl)
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
  python$PYTHON_VERSION -m pip install -r ./lite/tests/unittest_py/requirements.txt
}

function run_test() {
  local target_list="Host,X86"
  local targets=(${target_list//,/ })
  rm -rf $(find $WORKSPACE/lite/tests/unittest_py/ -name statics_data)

  for target in ${targets[@]}; do
    auto_scan_test $target
  done
}

function get_summary() {
  cd $WORKSPACE/lite/tests/unittest_py/op/
  python$PYTHON_VERSION ../global_var_model.py
  cd $WORKSPACE/lite/tests/unittest_py/pass/
  python$PYTHON_VERSION ../global_var_model.py
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
  local target_list="x86"
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
  compile_publish_inference_lib

  # Run unittests
  run_test
}

function main() {
  # Parse command line.
  for i in "$@"; do
    case $i in
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
  pipeline
  get_summary

  # Run python demo
  run_python_demo

  # Uninstall paddlelite
  python$PYTHON_VERSION -m pip uninstall -y paddlelite

  echo "Success for targets: Host,X86"
}

main $@
