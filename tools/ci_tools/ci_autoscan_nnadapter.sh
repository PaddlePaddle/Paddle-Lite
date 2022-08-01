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
NNADAPTER_CAMBRICON_MLU_SDK_ROOT="/usr/local/neuware"
NNADAPTER_NVIDIA_CUDA_ROOT="/usr/local/cuda"
NNADAPTER_NVIDIA_TENSORRT_ROOT="/usr/local/tensorrt"
NNADAPTER_INTEL_OPENVINO_SDK_ROOT="/opt/intel/openvino_2022"
# Python version
PYTHON_VERSION=3.7
# Absolute path of Paddle-Lite source code.
SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
WORKSPACE=${SHELL_FOLDER%tools/ci_tools*}
# Skip op or pass, use | to separate them, such as "expand_op" or "expand_op|abc_pass", etc.
SKIP_LIST="abc_op|abc_pass"
UNIT_TEST_FILTER_TYPE=2 # 0: black list 1: white list
UNIT_TEST_CHECK_LIST=""
# Models URL
MODELS_URL="http://paddle-inference-dist.bj.bcebos.com/AI-Rank/mobile/MobileNetV1.tar.gz"

# Helper functions
source ${SHELL_FOLDER}/utils.sh

function auto_scan_test {
  rm -rf $(find $WORKSPACE/lite/tests/unittest_py/ -name statics_data)
  cd $WORKSPACE/lite/tests/unittest_py/op/
  unittests=$(ls | egrep -v $SKIP_LIST)
  local unit_test_check_items=(${UNIT_TEST_CHECK_LIST//,/ })
  for test in ${unittests[@]}; do
    local is_matched=0
    for unit_test_check_item in ${unit_test_check_items[@]}; do
        if [[ "$unit_test_check_item" == "$test" ]]; then
            echo "$test on the checklist."
            is_matched=1
            break
        fi
    done
    # black list
    if [[ $is_matched -eq 1 && $UNIT_TEST_FILTER_TYPE -eq 0 ]]; then
        continue
    fi
    # white list
    if [[ $is_matched -eq 0 && $UNIT_TEST_FILTER_TYPE -eq 1 ]]; then
        continue
    fi
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
  case $NNADAPTER_DEVICE_NAMES in
      "kunlunxin_xtcl")
          cmd_line="./lite/tools/build_linux.sh --arch=$ARCH --with_nnadapter=ON --nnadapter_with_kunlunxin_xtcl=ON  --nnadapter_kunlunxin_xtcl_sdk_url=$NNADAPTER_KUNLUNXIN_XTCL_SDK_URL --with_python=ON --python_version=$PYTHON_VERSION"
          ;;
      "cambricon_mlu")
          cmd_line="./lite/tools/build_linux.sh --arch=$ARCH --with_nnadapter=ON --nnadapter_with_cambricon_mlu=ON  --nnadapter_cambricon_mlu_sdk_root=$NNADAPTER_CAMBRICON_MLU_SDK_ROOT --with_python=ON --python_version=$PYTHON_VERSION --with_extra=ON --with_log=ON --with_exception=ON full_publish"
          ;;
      "nvidia_tensorrt")
          cmd_line="./lite/tools/build_linux.sh --arch=$ARCH --with_nnadapter=ON --nnadapter_with_nvidia_tensorrt=ON  --nnadapter_nvidia_cuda_root=$NNADAPTER_NVIDIA_CUDA_ROOT --nnadapter_nvidia_tensorrt_root=$NNADAPTER_NVIDIA_TENSORRT_ROOT --with_python=ON --python_version=$PYTHON_VERSION --with_extra=ON --with_log=ON --with_exception=ON full_publish"
          ;;
      "intel_openvino")
          cmd_line="./lite/tools/build_linux.sh --arch=$ARCH --with_nnadapter=ON --nnadapter_with_intel_openvino=ON --nnadapter_intel_openvino_sdk_root=$NNADAPTER_INTEL_OPENVINO_SDK_ROOT --with_python=ON --python_version=$PYTHON_VERSION --with_extra=ON --with_log=ON --with_exception=ON full_publish"
          ;;
      *)
          echo "NNADAPTER_DEVICE_NAMES=$NNADAPTER_DEVICE_NAMES is not support!"
          exit 1
  esac  
  
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
  if [ $NNADAPTER_DEVICE_NAMES != "intel_openvino" ]; then
    run_python_demo
  fi

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
          --unit_test_check_list=*)
              UNIT_TEST_CHECK_LIST="${i#*=}"
              shift
              ;;
          --unit_test_filter_type=*)
              UNIT_TEST_FILTER_TYPE="${i#*=}"
              shift
              ;;
          --python_version=*)
              PYTHON_VERSION="${i#*=}"
              shift
              ;;
          --nnadapter_with_kunlunxin_xtcl=*)
              NNADAPTER_DEVICE_NAMES="kunlunxin_xtcl"
              shift
              ;;
          --nnadapter_with_cambricon_mlu=*)
              NNADAPTER_DEVICE_NAMES="cambricon_mlu"
              shift
              ;;
          --nnadapter_with_nvidia_tensorrt=*)
              NNADAPTER_DEVICE_NAMES="nvidia_tensorrt"
              shift
              ;;
          --nnadapter_with_intel_openvino=*)
              NNADAPTER_DEVICE_NAMES="intel_openvino"
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
          --nnadapter_cambricon_mlu_sdk_root=*)
              NNADAPTER_CAMBRICON_MLU_SDK_ROOT="${i#*=}"
              shift
              ;;
          --nnadapter_nvidia_cuda_root=*)
              NNADAPTER_NVIDIA_CUDA_ROOT="${i#*=}"
              shift
              ;;
          --nnadapter_nvidia_tensorrt_root=*)
              NNADAPTER_NVIDIA_TENSORRT_ROOT="${i#*=}"
              shift
              ;;
          --nnadapter_intel_openvino_sdk_root*)
              NNADAPTER_INTEL_OPENVINO_SDK_ROOT="${i#*=}"
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
