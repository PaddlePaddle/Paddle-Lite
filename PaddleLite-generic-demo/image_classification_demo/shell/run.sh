#!/bin/bash
#MODEL_NAME=mobilenet_v1_fp32_224
#MODEL_NAME=mobilenet_v1_int8_224_per_layer
#MODEL_NAME=mobilenet_v1_int8_224_per_channel
#MODEL_NAME=mobilenet_v2_int8_224_per_layer
MODEL_NAME=resnet50_fp32_224
#MODEL_NAME=resnet50_int8_224_per_layer
#MODEL_NAME=shufflenet_v2_int8_224_per_layer
if [ -n "$1" ]; then
  MODEL_NAME=$1
fi
if [ ! -d "../assets/models/$MODEL_NAME" ];then
  MODEL_URL="http://paddlelite-demo.bj.bcebos.com/devices/generic/models/${MODEL_NAME}.tar.gz"
  echo "Model $MODEL_NAME not found! Try to download it from $MODEL_URL ..."
  curl $MODEL_URL -o -| tar -xz -C ../assets/models
  if [[ $? -ne 0 ]]; then
    echo "Model $MODEL_NAME download failed!"
    exit 1
  fi
fi

CONFIG_NAME=imagenet_224.txt
if [ -n "$2" ]; then
  CONFIG_NAME=$2
fi

DATASET_NAME=test
if [ -n "$3" ]; then
  DATASET_NAME=$3
fi

# For TARGET_OS=android, TARGET_ABI should be arm64-v8a or armeabi-v7a.
# For TARGET_OS=linux, TARGET_ABI should be arm64, armhf or amd64.
# Kirin810/820/985/990/9000/9000E: TARGET_OS=android and TARGET_ABI=arm64-v8a
# MT8168/8175, Kirin810/820/985/990/9000/9000E: TARGET_OS=android and TARGET_ABI=armeabi-v7a
# RK1808EVB, TB-RK1808S0, Kunpeng-920+Ascend310: TARGET_OS=linux and TARGET_ABI=arm64
# RK1806EVB, RV1109/1126 EVB: TARGET_OS=linux and TARGET_ABI=armhf 
# Intel-x86+Ascend310: TARGET_OS=linux and TARGET_ABI=amd64
# Intel-x86+CambriconMLU: TARGET_OS=linux and TARGET_ABI=amd64
TARGET_OS=linux
if [ -n "$4" ]; then
  TARGET_OS=$4
fi

TARGET_ABI=loongarch64
if [ -n "$5" ]; then
  TARGET_ABI=$5
fi

# RK1808EVB, TB-RK1808S0, RK1806EVB, RV1109/1126 EVB: NNADAPTER_DEVICE_NAMES=rockchip_npu
# MT8168/8175: NNADAPTER_DEVICE_NAMES=mediatek_apu
# Kirin810/820/985/990/9000/9000E: NNADAPTER_DEVICE_NAMES=huawei_kirin_npu
# Ascend310: NNADAPTER_DEVICE_NAMES=huawei_ascend_npu
# CambriconMLU: NNADAPTER_DEVICE_NAMES=cambricon_mlu
# CPU only: NNADAPTER_DEVICE_NAMES=cpu
NNADAPTER_DEVICE_NAMES="opencl"
if [ -n "$6" ]; then
  NNADAPTER_DEVICE_NAMES="$6"
fi
NNADAPTER_DEVICE_NAMES_LIST=(${NNADAPTER_DEVICE_NAMES//,/ })
NNADAPTER_DEVICE_NAMES_TEXT=${NNADAPTER_DEVICE_NAMES//,/_}

if [ -n "$7" ] && [ "$7" != "null" ]; then
  NNADAPTER_CONTEXT_PROPERTIES="$7"
fi

NNADAPTER_MODEL_CACHE_DIR="null"
if [ -n "$8" ]; then
  NNADAPTER_MODEL_CACHE_DIR="$8"
fi

NNADAPTER_MODEL_CACHE_TOKEN="null"
if [ -n "$9" ]; then
  NNADAPTER_MODEL_CACHE_TOKEN="$9"
fi

#NNADAPTER_SUBGRAPH_PARTITION_CONFIG_PATH="null"
NNADAPTER_SUBGRAPH_PARTITION_CONFIG_PATH="../assets/models/$MODEL_NAME/${NNADAPTER_DEVICE_NAMES_TEXT}_subgraph_partition_config_file.txt"

#NNADAPTER_MIXED_PRECISION_QUANTIZATION_CONFIG_PATH="null"
NNADAPTER_MIXED_PRECISION_QUANTIZATION_CONFIG_PATH="../assets/models/$MODEL_NAME/${NNADAPTER_DEVICE_NAMES_TEXT}_mixed_precision_quantization_config_file.txt"

export GLOG_v=5
export SUBGRAPH_ONLINE_MODE=true
if [[ "$NNADAPTER_DEVICE_NAMES" =~ "rockchip_npu" ]]; then
  export RKNPU_LOGLEVEL=5
  export RKNN_LOG_LEVEL=5
  ulimit -c unlimited
  echo userspace > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor
  echo $(cat /sys/devices/system/cpu/cpufreq/policy0/scaling_max_freq) > /sys/devices/system/cpu/cpufreq/policy0/scaling_setspeed
fi

if [[ "$NNADAPTER_DEVICE_NAMES" =~ "amlogic_npu" ]]; then
  echo performance > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor
fi

if [[ "$NNADAPTER_DEVICE_NAMES" =~ "imagination_nna" ]]; then
  echo performance > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor
  echo performance > /sys/devices/system/cpu/cpufreq/policy4/scaling_governor
fi

if [[ "$NNADAPTER_DEVICE_NAMES" =~ "verisilicon_timvx" ]]; then
  export VIV_VX_ENABLE_GRAPH_TRANSFORM=-pcq:1
  export VIV_VX_SET_PER_CHANNEL_ENTROPY=100
  export VSI_NN_LOG_LEVEL=5
fi

export LD_LIBRARY_PATH=../../libs/PaddleLite/$TARGET_OS/$TARGET_ABI/lib/cpu:../../libs/PaddleLite/$TARGET_OS/$TARGET_ABI/lib:.:$LD_LIBRARY_PATH
for NNADAPTER_DEVICE_NAME in ${NNADAPTER_DEVICE_NAMES_LIST[@]}
do
  export LD_LIBRARY_PATH=../../libs/PaddleLite/$TARGET_OS/$TARGET_ABI/lib/$NNADAPTER_DEVICE_NAME:$LD_LIBRARY_PATH
done
if [[ "$NNADAPTER_DEVICE_NAMES" =~ "huawei_ascend_npu" ]]; then
  HUAWEI_ASCEND_TOOLKIT_HOME="/usr/local/Ascend/ascend-toolkit/latest"
  if [ "$TARGET_OS" == "linux" ]; then
    if [[ "$TARGET_ABI" != "arm64" && "$TARGET_ABI" != "amd64" ]]; then
      echo "Unknown OS $TARGET_OS, only supports 'arm64' or 'amd64' for Huawei Ascend NPU."
      exit -1
    fi
  else
    echo "Unknown OS $TARGET_OS, only supports 'linux' for Huawei Ascend NPU."
    exit -1
  fi
  if [[ ! "$NNADAPTER_CONTEXT_PROPERTIES" =~ "HUAWEI_ASCEND_NPU_SELECTED_DEVICE_IDS" ]]; then
    NNADAPTER_CONTEXT_PROPERTIES="HUAWEI_ASCEND_NPU_SELECTED_DEVICE_IDS=0;${NNADAPTER_CONTEXT_PROPERTIES}"
  fi
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/stub:$HUAWEI_ASCEND_TOOLKIT_HOME/acllib/lib64:$HUAWEI_ASCEND_TOOLKIT_HOME/atc/lib64:$HUAWEI_ASCEND_TOOLKIT_HOME/opp/op_proto/built-in
  export PYTHONPATH=$PYTHONPATH:$HUAWEI_ASCEND_TOOLKIT_HOME/fwkacllib/python/site-packages:$HUAWEI_ASCEND_TOOLKIT_HOME/acllib/python/site-packages:$HUAWEI_ASCEND_TOOLKIT_HOME/toolkit/python/site-packages:$HUAWEI_ASCEND_TOOLKIT_HOME/atc/python/site-packages:$HUAWEI_ASCEND_TOOLKIT_HOME/pyACL/python/site-packages/acl
  export PATH=$PATH:$HUAWEI_ASCEND_TOOLKIT_HOME/atc/ccec_compiler/bin:${HUAWEI_ASCEND_TOOLKIT_HOME}/acllib/bin:$HUAWEI_ASCEND_TOOLKIT_HOME/atc/bin
  export ASCEND_AICPU_PATH=$HUAWEI_ASCEND_TOOLKIT_HOME
  export ASCEND_OPP_PATH=$HUAWEI_ASCEND_TOOLKIT_HOME/opp
  export TOOLCHAIN_HOME=$HUAWEI_ASCEND_TOOLKIT_HOME/toolkit
  export ASCEND_SLOG_PRINT_TO_STDOUT=1
  export ASCEND_GLOBAL_LOG_LEVEL=3
fi

if [[ "$NNADAPTER_DEVICE_NAMES" =~ "kunlunxin_xtcl" ]]; then
  export XTCL_AUTO_ALLOC_L3=1
  export XTCL_CONV_USE_FP16=1
  export XTCL_QUANTIZE_WEIGHT=1
  export XTCL_L3_SIZE=16777216
fi

if [[ "$NNADAPTER_DEVICE_NAMES" =~ "cambricon_mlu" ]]; then
  export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/neuware/lib64"
fi

if [[ "$NNADAPTER_DEVICE_NAMES" =~ "intel_openvino" ]]; then
  if [[ ! "$NNADAPTER_CONTEXT_PROPERTIES" =~ "INTEL_OPENVINO_SELECT_DEVICE_NAMES" ]]; then
    NNADAPTER_CONTEXT_PROPERTIES="INTEL_OPENVINO_SELECT_DEVICE_NAMES=CPU;${NNADAPTER_CONTEXT_PROPERTIES}"
  fi
fi

if [[ "$NNADAPTER_DEVICE_NAMES" =~ "qualcomm_qnn" ]]; then
  export ADSP_LIBRARY_PATH="../../libs/PaddleLite/$TARGET_OS/$TARGET_ABI/lib/qualcomm_qnn/hexagon-v68/lib/unsigned"
  if [[ ! "$NNADAPTER_CONTEXT_PROPERTIES" =~ "QUALCOMM_QNN_DEVICE_TYPE" ]]; then
    NNADAPTER_CONTEXT_PROPERTIES="QUALCOMM_QNN_DEVICE_TYPE=HTP;${NNADAPTER_CONTEXT_PROPERTIES}"
  fi
fi

if [[ "$NNADAPTER_DEVICE_NAMES" =~ "xpu" ]]; then
  export XPU_VISIBLE_DEVICES=0
fi

if [ -z "$NNADAPTER_CONTEXT_PROPERTIES" ]; then
  NNADAPTER_CONTEXT_PROPERTIES="null"
fi

BUILD_DIR=build.${TARGET_OS}.${TARGET_ABI}

set -e
if [ "$NNADAPTER_MODEL_CACHE_DIR" != "null" ]; then
  NNADAPTER_MODEL_CACHE_DIR=../assets/models/$NNADAPTER_MODEL_CACHE_DIR
  mkdir -p $NNADAPTER_MODEL_CACHE_DIR
fi
chmod +x ./$BUILD_DIR/demo
./$BUILD_DIR/demo ../assets/models/$MODEL_NAME ../assets/configs/$CONFIG_NAME ../assets/datasets/$DATASET_NAME $NNADAPTER_DEVICE_NAMES "$NNADAPTER_CONTEXT_PROPERTIES" $NNADAPTER_MODEL_CACHE_DIR $NNADAPTER_MODEL_CACHE_TOKEN $NNADAPTER_SUBGRAPH_PARTITION_CONFIG_PATH $NNADAPTER_MIXED_PRECISION_QUANTIZATION_CONFIG_PATH
