#!/bin/bash
#MODEL_NAME=conv_add_relu_dwconv_add_relu_224_int8_per_layer
#MODEL_NAME=conv_bn_relu_224_int8_per_channel
#MODEL_NAME=conv_bn_relu_dwconv_bn_relu_224_int8_per_channel
#MODEL_NAME=conv_add_relu_dwconv_add_relu_x4_224_int8_per_channel
#MODEL_NAME=conv_add_relu_dwconv_add_relu_x27_pool2d_224_int8_per_channel
#MODEL_NAME=conv_add_relu_dwconv_add_relu_x27_pool2d_mul_add_224_int8_per_channel
#MODEL_NAME=conv_add_relu_dwconv_add_relu_conv_add_relu_dwconv_add_relu_224_int8_per_channel
MODEL_NAME=conv_bn_relu_224_fp32
#MODEL_NAME=conv_bn_relu_dwconv_bn_relu_224_fp32
#MODEL_NAME=conv_bn_relu_dwconv_bn_relu_x27_224_fp32
#MODEL_NAME=conv_bn_relu_dwconv_bn_relu_x27_pool2d_224_fp32
#MODEL_NAME=conv_bn_relu_dwconv_bn_relu_x27_pool2d_mul_add_224_fp32
#MODEL_NAME=conv_bn_relu_pool2d_224_fp32
#MODEL_NAME=conv_bn_relu_pool2d_res2a_224_fp32
#MODEL_NAME=conv_bn_relu_pool2d_res2a_res2b_224_fp32
#MODEL_NAME=conv_bn_relu_pool2d_res2a_res2b_res2c_224_fp32
INPUT_SHAPES="1,3,224,224"
INPUT_TYPES="float32"
OUTPUT_TYPES="float32"

#MODEL_NAME=conv_add_144_192_int8_per_layer
#MODEL_NAME=conv_add_scale_144_192_int8_per_layer
#MODEL_NAME=conv_add_scale_relu6_144_192_int8_per_layer
#MODEL_NAME=conv_add_scale_relu6_mul_144_192_int8_per_layer
#MODEL_NAME=conv_add_scale_sigmoid_144_192_int8_per_layer
#MODEL_NAME=conv_add_scale_sigmoid_relu_144_192_int8_per_layer
#MODEL_NAME=conv_add_scale_sigmoid_relu_mul_144_192_int8_per_layer
#INPUT_SHAPES="1,3,192,144"
#INPUT_TYPES="float32"
#OUTPUT_TYPES="float32"

#MODEL_NAME=eltwise_mul_broadcast_per_layer
#INPUT_SHAPES="1,3,384,384"
#INPUT_TYPES="float32"
#OUTPUT_TYPES="float32"

#MODEL_NAME=dwconv_ic_128_groups_128_oc_256_per_layer
#INPUT_SHAPES="1,3,320,320"
#INPUT_TYPES="float32"
#OUTPUT_TYPES="float32"

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

if [ -n "$2" ]; then
  INPUT_SHAPES=$2
fi

if [ -n "$3" ]; then
  INPUT_TYPES=$3
fi

if [ -n "$4" ]; then
  OUTPUT_TYPES=$4
fi

WORK_SPACE=/data/local/tmp/test

# For TARGET_OS=android, TARGET_ABI should be arm64-v8a or armeabi-v7a.
# For TARGET_OS=linux, TARGET_ABI should be arm64, armhf or amd64.
# Kirin810/820/985/990/9000/9000E: TARGET_OS=android and TARGET_ABI=arm64-v8a
# MT8168/8175, Kirin810/820/985/990/9000/9000E: TARGET_OS=android and TARGET_ABI=armeabi-v7a
# RK1808EVB, TB-RK1808S0: TARGET_OS=linux and TARGET_ABI=arm64
# RK1806EVB, RV1109/1126 EVB: TARGET_OS=linux and TARGET_ABI=armhf
TARGET_OS=android
if [ -n "$5" ]; then
  TARGET_OS=$5
fi

if [ "$TARGET_OS" == "linux" ]; then
  WORK_SPACE=/var/tmp/test
fi

TARGET_ABI=arm64-v8a
if [ -n "$6" ]; then
  TARGET_ABI=$6
fi

# RK1808EVB, TB-RK1808S0, RK1806EVB, RV1109/1126 EVB: NNADAPTER_DEVICE_NAMES=rockchip_npu
# MT8168/8175: NNADAPTER_DEVICE_NAMES=mediatek_apu
# Kirin810/820/985/990/9000/9000E: NNADAPTER_DEVICE_NAMES=huawei_kirin_npu
# CPU only: NNADAPTER_DEVICE_NAMES=cpu
NNADAPTER_DEVICE_NAMES="cpu"
if [ -n "$7" ]; then
  NNADAPTER_DEVICE_NAMES="$7"
fi
NNADAPTER_DEVICE_NAMES_LIST=(${NNADAPTER_DEVICE_NAMES//,/ })
NNADAPTER_DEVICE_NAMES_TEXT=${NNADAPTER_DEVICE_NAMES//,/_}

ADB_DEVICE_NAME=
if [ -n "$8" ]; then
  ADB_DEVICE_NAME="-s $8"
fi

if [ -n "$9" ] && [ "$9" != "null" ]; then
  NNADAPTER_CONTEXT_PROPERTIES="$9"
fi

NNADAPTER_MODEL_CACHE_DIR="null"
if [ -n "${10}" ]; then
  NNADAPTER_MODEL_CACHE_DIR="${10}"
fi

NNADAPTER_MODEL_CACHE_TOKEN="null"
if [ -n "${11}" ]; then
  NNADAPTER_MODEL_CACHE_TOKEN="${11}"
fi

#NNADAPTER_SUBGRAPH_PARTITION_CONFIG_PATH="null"
NNADAPTER_SUBGRAPH_PARTITION_CONFIG_PATH="./$MODEL_NAME/subgraph_partition_config_file.txt"

#NNADAPTER_MIXED_PRECISION_QUANTIZATION_CONFIG_PATH="null"
NNADAPTER_MIXED_PRECISION_QUANTIZATION_CONFIG_PATH="./$MODEL_NAME/mixed_precision_quantization_config_file.txt"

EXPORT_ENVIRONMENT_VARIABLES="export GLOG_v=5; export SUBGRAPH_ONLINE_MODE=true;"
if [[ "$NNADAPTER_DEVICE_NAMES" =~ "rockchip_npu" ]]; then
  EXPORT_ENVIRONMENT_VARIABLES="${EXPORT_ENVIRONMENT_VARIABLES}export RKNPU_LOGLEVEL=5; export RKNN_LOG_LEVEL=5; ulimit -c unlimited;"
  adb $ADB_DEVICE_NAME shell "echo userspace > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor"
  adb $ADB_DEVICE_NAME shell "echo $(cat /sys/devices/system/cpu/cpufreq/policy0/scaling_max_freq) > /sys/devices/system/cpu/cpufreq/policy0/scaling_setspeed"
fi

if [[ "$NNADAPTER_DEVICE_NAMES" =~ "qualcomm_qnn" ]]; then
  EXPORT_ENVIRONMENT_VARIABLES="${EXPORT_ENVIRONMENT_VARIABLES}export ADSP_LIBRARY_PATH=$WORK_SPACE/qualcomm_qnn/hexagon-v68/lib/unsigned;"
  if [[ ! "$NNADAPTER_CONTEXT_PROPERTIES" =~ "QUALCOMM_QNN_DEVICE_TYPE" ]]; then
    NNADAPTER_CONTEXT_PROPERTIES="QUALCOMM_QNN_DEVICE_TYPE=HTP;${NNADAPTER_CONTEXT_PROPERTIES}"
  fi
fi

if [[ "$NNADAPTER_DEVICE_NAMES" =~ "verisilicon_timvx" ]]; then
  EXPORT_ENVIRONMENT_VARIABLES="${EXPORT_ENVIRONMENT_VARIABLES}export VIV_VX_ENABLE_GRAPH_TRANSFORM=-pcq:1; export VIV_VX_SET_PER_CHANNEL_ENTROPY=100; export VSI_NN_LOG_LEVEL=5;"
fi

EXPORT_ENVIRONMENT_VARIABLES="${EXPORT_ENVIRONMENT_VARIABLES}export LD_LIBRARY_PATH="
for NNADAPTER_DEVICE_NAME in ${NNADAPTER_DEVICE_NAMES_LIST[@]}
do
  EXPORT_ENVIRONMENT_VARIABLES="$EXPORT_ENVIRONMENT_VARIABLES./$NNADAPTER_DEVICE_NAME:"
done
EXPORT_ENVIRONMENT_VARIABLES="$EXPORT_ENVIRONMENT_VARIABLES./cpu:.:\$LD_LIBRARY_PATH;"

if [ -z "$NNADAPTER_CONTEXT_PROPERTIES" ]; then
  NNADAPTER_CONTEXT_PROPERTIES="null"
fi

BUILD_DIR=build.${TARGET_OS}.${TARGET_ABI}

set -e
adb $ADB_DEVICE_NAME shell "rm -rf $WORK_SPACE"
adb $ADB_DEVICE_NAME shell "mkdir -p $WORK_SPACE"
adb $ADB_DEVICE_NAME push ../../libs/PaddleLite/$TARGET_OS/$TARGET_ABI/lib/libpaddle_*.so $WORK_SPACE
for NNADAPTER_DEVICE_NAME in ${NNADAPTER_DEVICE_NAMES_LIST[@]}
do
  adb $ADB_DEVICE_NAME push ../../libs/PaddleLite/$TARGET_OS/$TARGET_ABI/lib/$NNADAPTER_DEVICE_NAME $WORK_SPACE
done
adb $ADB_DEVICE_NAME push ../../libs/PaddleLite/$TARGET_OS/$TARGET_ABI/lib/cpu $WORK_SPACE
adb $ADB_DEVICE_NAME push ../assets/models/$MODEL_NAME $WORK_SPACE
set +e
adb $ADB_DEVICE_NAME push ../assets/models/${MODEL_NAME}.nb $WORK_SPACE
if [ "$NNADAPTER_MODEL_CACHE_DIR" != "null" ]; then
  adb $ADB_DEVICE_NAME push ../assets/models/$NNADAPTER_MODEL_CACHE_DIR $WORK_SPACE
  adb $ADB_DEVICE_NAME shell "mkdir -p $WORK_SPACE/$NNADAPTER_MODEL_CACHE_DIR"
fi
set -e
adb $ADB_DEVICE_NAME push $BUILD_DIR/demo $WORK_SPACE
adb $ADB_DEVICE_NAME shell "cd $WORK_SPACE; ${EXPORT_ENVIRONMENT_VARIABLES} chmod +x ./demo; ./demo $MODEL_NAME $INPUT_SHAPES $INPUT_TYPES $OUTPUT_TYPES $NNADAPTER_DEVICE_NAMES \"$NNADAPTER_CONTEXT_PROPERTIES\" $NNADAPTER_MODEL_CACHE_DIR $NNADAPTER_MODEL_CACHE_TOKEN $NNADAPTER_SUBGRAPH_PARTITION_CONFIG_PATH $NNADAPTER_MIXED_PRECISION_QUANTIZATION_CONFIG_PATH"
adb $ADB_DEVICE_NAME pull $WORK_SPACE/${MODEL_NAME}.nb ../assets/models/
if [ "$NNADAPTER_MODEL_CACHE_DIR" != "null" ]; then
  adb $ADB_DEVICE_NAME pull $WORK_SPACE/$NNADAPTER_MODEL_CACHE_DIR ../assets/models/
fi
