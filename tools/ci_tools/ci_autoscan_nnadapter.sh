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
# Python version
PYTHON_VERSION=3.7
# Absolute path of Paddle-Lite source code.
SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
WORKSPACE=${SHELL_FOLDER%tools/ci_tools*}
# Skip op or pass, use | to separate them, such as "expand_op" or "expand_op|abc_pass", etc.
SKIP_LIST="abc_op|abc_pass"
# white list
cambricon_mlu_unittests_white_list="test_softmax_op.py | test_batch_norm_op.py | test_reshape2_op.py |
  test_reshape_op.py | test_elementwise_add_op.py |test_elementwise_sub_op.py | test_elementwise_mul_op.py |
  test_pow_op.py | test_sigmoid_op.py | test_relu_op.py | test_relu6_op.py | test_leaky_relu_op.py |
  test_tanh_op.py | test_log_op.py | test_equal_op.py | test_scale_op.py"

cambricon_mlu_unittests_black_list="test_cast_op.py | test_clip_op.py |  test_conv2d_op.py | test_conv2d_transpose_op |
  test_deformable_conv_op.py | test_pool2d_op.py | test_unsqueeze2_op.py | test_unsqueeze_op.py |
  test_expand_v2_op.py | test_compare_less_op | test_greater_op.py | test_reduce_mean_op.py |
  test_shape_op.py | test_slice_op.py test_squeeze_op.py | test_squeeze2_op.py |
  test_fill_constant_op.py | test_fill_any_like_op.py | test_concat_op.py |
  test_nearest_interp_op.py | test_nearest_interp_v2_op.py | test_bilinear_interp_op.py |test_bilinear_interp_v2_op.py |
  test_flatten_contiguous_range_op.py | test_flatten_op.py | test_flatten_v2_op.py |
  test_fc_op.py | test_norm_op.py | test_gather_op.py | test_elementwise_div_op.py"

# Models URL
MODELS_URL="https://paddle-inference-dist.bj.bcebos.com/AI-Rank/mobile/MobileNetV1.tar.gz"

# Helper functions
source ${SHELL_FOLDER}/utils.sh

function auto_scan_test {
  rm -rf $(find $WORKSPACE/lite/tests/unittest_py/ -name statics_data)
  cd $WORKSPACE/lite/tests/unittest_py/op/
  unittests=$(ls | egrep -v $SKIP_LIST)
  
  if [[ $NNADAPTER_DEVICE_NAMES == "cambricon_mlu" ]]; then
    unittests=$cambricon_mlu_unittests_white_list
  fi
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
  case $NNADAPTER_DEVICE_NAMES in
      "kunlunxin_xtcl")
          cmd_line="./lite/tools/build_linux.sh --arch=$ARCH --with_nnadapter=ON --nnadapter_with_kunlunxin_xtcl=ON  --nnadapter_kunlunxin_xtcl_sdk_url=$NNADAPTER_KUNLUNXIN_XTCL_SDK_URL --with_python=ON --python_version=$PYTHON_VERSION"
          ;;
      "cambricon_mlu")
          cmd_line="./lite/tools/build_linux.sh --arch=$ARCH --with_nnadapter=ON --nnadapter_with_cambricon_mlu=ON  --nnadapter_cambricon_mlu_sdk_root=$NNADAPTER_CAMBRICON_MLU_SDK_ROOT --with_python=ON --python_version=$PYTHON_VERSION --with_extra=ON --with_log=ON --with_exception=ON full_publish"
          ;;
      *)
          echo "NNADAPTER_DEVICE_NAMES=$NNADAPTER_DEVICE_NAMES is not support!"
          exit 1
  esac  
  
  # $cmd_line

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
          --nnadapter_with_cambricon_mlu=*)
              NNADAPTER_WITH_CAMBRICON_MLU="${i#*=}"
              NNADAPTER_DEVICE_NAMES="cambricon_mlu"
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
