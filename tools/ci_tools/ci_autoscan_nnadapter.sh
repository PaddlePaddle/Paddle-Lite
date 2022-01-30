#!/bin/bash
# Start the CI task of unittest for op and pass.
set -ex

#####################################################################################################
# 1. global variables, you can change them according to your requirements
#####################################################################################################
OS="android"
ARCH="armv8"
TOOLCHAIN="gcc"
NNADAPTER_DEVICE_NAMES=""
# Python version
PYTHON_VERSION=3.7
# Absolute path of Paddle-Lite source code.
SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
WORKSPACE=${SHELL_FOLDER%tools/ci_tools*}
# Skip op or pass, use | to separate them, such as "expand_op" or "expand_op|abc_pass", etc.
SKIP_LIST="abc_op|abc_pass"
# Models URL
MODELS_URL="https://paddle-inference-dist.bj.bcebos.com/AI-Rank/mobile/MobileNetV1.tar.gz"

# Helper functions
source ${SHELL_FOLDER}/utils.sh

function auto_scan_test {
  rm -rf $(find $WORKSPACE/lite/tests/unittest_py/ -name statics_data)
  cd $WORKSPACE/lite/tests/unittest_py/op/
  unittests=$(ls | egrep -v $SKIP_LIST)
  for test in ${unittests[@]}; do
    if [[ "$test" =~ py$ ]]; then
      python$PYTHON_VERSION $test --target=NNAdapter --nnadapter_device_names=$NNADAPTER_DEVICE_NAMES
    fi
  done
}

function get_summary() {
  cd $WORKSPACE/lite/tests/unittest_py/op/
  python$PYTHON_VERSION ../global_var_model.py
}

function check_classification_result() {
  local target=$1
  local nnadapter_device_names=$2
  local log_file=$3
  local result_class_name="Egyptian cat"

  local ret=$(grep "$result_class_name" $log_file)
  if [[ -z "$ret" ]]; then
    echo "Wrong result on $target.$nnadapter_device_names exit!"
    exit 1
  fi
}

function run_python_demo() {
  local target_list="NNAdapter"
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
        --backend ${target} \
        --nnadapter_device_names $NNADAPTER_DEVICE_NAMES 2>&1 | tee $log_file
    check_classification_result $target $NNADAPTER_DEVICE_NAMES $log_file
  done
}

function build_and_test {
  cd $WORKSPACE
  # Step1. Compiling python installer
  local cmd_line=""
  if [ "$NNADAPTER_DEVICE_NAMES" = "kunlunxin_xtcl" ]; then 
      cmd_line="./lite/tools/build_linux.sh --arch=$ARCH --with_nnadapter=ON --nnadapter_with_kunlunxin_xtcl=ON  --nnadapter_kunlunxin_xtcl_sdk_url=$NNADAPTER_KUNLUNXIN_XTCL_SDK_URL --with_python=ON --python_version=$PYTHON_VERSION"
  else
      echo "NNADAPTER_DEVICE_NAMES=$NNADAPTER_DEVICE_NAMES is not support!"
      exit 1
  fi
  $cmd_line

  # Step2. Checking results: cplus and python inference lib
  local whl_path=$(find ./build.lite.linux.$ARCH.$TOOLCHAIN/* -name *whl)
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

  # Step4. # Run op/pass unittests
  auto_scan_test
  get_summary

  # Step5. run_python_demo
  run_python_demo

  # Uninstall paddlelite
  python$PYTHON_VERSION -m pip uninstall -y paddlelite
}

function main() {
  for i in "$@"; do
      case $i in
          --os=*)
              OS="${i#*=}"
              shift
              ;;
          --arch=*)
              ARCH="${i#*=}"
              shift
              ;;
          --toolchain=*)
              TOOLCHAIN="${i#*=}"
              shift
              ;;
          --skip_list=*)
              SKIP_LIST="${i#*=}"
              shift
              ;;
          --python_version=*)
              PYTHON_VERSION="${i#*=}"
              shift
              ;;
          --nnadapter_with_kunlunxin_xtcl=*)
              NNADAPTER_WITH_KUNLUNXIN_XTCL="${i#*=}"
              NNADAPTER_DEVICE_NAMES="kunlunxin_xtcl"
              shift
              ;;
          --nnadapter_kunlunxin_xtcl_sdk_root=*)
              NNADAPTER_KUNLUNXIN_XTCL_SDK_ROOT="${i#*=}"
              shift
              ;;
          --nnadapter_kunlunxin_xtcl_sdk_url=*)
              NNADAPTER_KUNLUNXIN_XTCL_SDK_URL="${i#*=}"
              shift
              ;;
          --nnadapter_kunlunxin_xtcl_sdk_env=*)
              NNADAPTER_KUNLUNXIN_XTCL_SDK_ENV="${i#*=}"
              shift
              ;;
          *)
              echo "$i"
              echo "Unknown option, exit"
              exit 1
              ;;
      esac
  done

  build_and_test
  echo "Success for targets: NNAdapter, nnadapter_device_names=$NNADAPTER_DEVICE_NAMES"
}

main $@
