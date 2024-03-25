#!/bin/bash
MODEL_NAME=pp_liteseg_stdc1_cityscapes_1024x512_scale_1_0_160k_with_argmax_fp32_512_1024
#MODEL_NAME=pp_liteseg_stdc1_cityscapes_1024x512_scale_1_0_160k_with_argmax_int8_512_1024_per_layer
#MODEL_NAME=pp_liteseg_stdc1_cityscapes_1024x512_scale_1_0_160k_with_argmax_int8_512_1024_per_channel
#MODEL_NAME=portrait_pp_humansegv1_lite_398x224_with_softmax_fp32_224_398
#MODEL_NAME=portrait_pp_humansegv1_lite_398x224_with_softmax_int8_224_398_per_layer
#MODEL_NAME=portrait_pp_humansegv1_lite_398x224_with_softmax_int8_224_398_per_channel
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

CONFIG_NAME=cityscapes_512_1024.txt
#CONFIG_NAME=human_224_398.txt
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
TARGET_OS=android
if [ -n "$4" ]; then
  TARGET_OS=$4
fi

WORK_SPACE=/data/local/tmp/test
if [ "$TARGET_OS" == "linux" ]; then
  WORK_SPACE=/var/tmp/test
fi

TARGET_ABI=arm64-v8a
if [ -n "$5" ]; then
  TARGET_ABI=$5
fi

# RK1808EVB, TB-RK1808S0, RK1806EVB, RV1109/1126 EVB: NNADAPTER_DEVICE_NAMES=rockchip_npu
# MT8168/8175: NNADAPTER_DEVICE_NAMES=mediatek_apu
# Kirin810/820/985/990/9000/9000E: NNADAPTER_DEVICE_NAMES=huawei_kirin_npu
# CPU only: NNADAPTER_DEVICE_NAMES=cpu
NNADAPTER_DEVICE_NAMES="cpu"
if [ -n "$6" ]; then
  NNADAPTER_DEVICE_NAMES="$6"
fi
NNADAPTER_DEVICE_NAMES_LIST=(${NNADAPTER_DEVICE_NAMES//,/ })
NNADAPTER_DEVICE_NAMES_TEXT=${NNADAPTER_DEVICE_NAMES//,/_}

ADB_DEVICE_NAME=
if [ -n "$7" ]; then
  ADB_DEVICE_NAME="-s $7"
fi

if [ -n "$8" ] && [ "$8" != "null" ]; then
  NNADAPTER_CONTEXT_PROPERTIES="$8"
fi

NNADAPTER_MODEL_CACHE_DIR="null"
if [ -n "$9" ]; then
  NNADAPTER_MODEL_CACHE_DIR="$9"
fi

NNADAPTER_MODEL_CACHE_TOKEN="null"
if [ -n "${10}" ]; then
  NNADAPTER_MODEL_CACHE_TOKEN="${10}"
fi

#NNADAPTER_SUBGRAPH_PARTITION_CONFIG_PATH="null"
NNADAPTER_SUBGRAPH_PARTITION_CONFIG_PATH="./$MODEL_NAME/${NNADAPTER_DEVICE_NAMES_TEXT}_subgraph_partition_config_file.txt"

#NNADAPTER_MIXED_PRECISION_QUANTIZATION_CONFIG_PATH="null"
NNADAPTER_MIXED_PRECISION_QUANTIZATION_CONFIG_PATH="./$MODEL_NAME/${NNADAPTER_DEVICE_NAMES_TEXT}_mixed_precision_quantization_config_file.txt"

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
adb $ADB_DEVICE_NAME push ../assets/configs/. $WORK_SPACE
adb $ADB_DEVICE_NAME push $BUILD_DIR/demo $WORK_SPACE
set +e
adb $ADB_DEVICE_NAME push ../assets/models/${MODEL_NAME}.nb $WORK_SPACE
if [ "$NNADAPTER_MODEL_CACHE_DIR" != "null" ]; then
  adb $ADB_DEVICE_NAME push ../assets/models/$NNADAPTER_MODEL_CACHE_DIR $WORK_SPACE
  adb $ADB_DEVICE_NAME shell "mkdir -p $WORK_SPACE/$NNADAPTER_MODEL_CACHE_DIR"
fi
set -e
COMMAND_LINE="cd $WORK_SPACE; $EXPORT_ENVIRONMENT_VARIABLES chmod +x ./demo; ./demo ./$MODEL_NAME ./$CONFIG_NAME ./$DATASET_NAME $NNADAPTER_DEVICE_NAMES \"$NNADAPTER_CONTEXT_PROPERTIES\" $NNADAPTER_MODEL_CACHE_DIR $NNADAPTER_MODEL_CACHE_TOKEN $NNADAPTER_SUBGRAPH_PARTITION_CONFIG_PATH $NNADAPTER_MIXED_PRECISION_QUANTIZATION_CONFIG_PATH"
rm -rf ../assets/datasets/$DATASET_NAME/outputs
mkdir -p ../assets/datasets/$DATASET_NAME/outputs
SPLIT_COUNT=200
SPLIT_INDEX=0
SAMPLE_INDEX=0
SAMPLE_START=0
for SAMPLE_NAME in $(cat ../assets/datasets/$DATASET_NAME/list.txt); do
  echo $SAMPLE_INDEX + ": " + $SAMPLE_NAME
  if [ $SAMPLE_INDEX -ge $SAMPLE_START ] ; then 
    if [ $SPLIT_INDEX -eq $SPLIT_COUNT ] ; then
      adb $ADB_DEVICE_NAME push list.txt $WORK_SPACE/$DATASET_NAME/
      adb $ADB_DEVICE_NAME shell "$COMMAND_LINE"
      adb $ADB_DEVICE_NAME pull $WORK_SPACE/$DATASET_NAME/outputs/ ../assets/datasets/$DATASET_NAME/outputs/
      SPLIT_INDEX=0
    fi
    if [ $SPLIT_INDEX -eq 0 ] ; then 
      adb $ADB_DEVICE_NAME shell "rm -rf $WORK_SPACE/$DATASET_NAME/inputs"
      adb $ADB_DEVICE_NAME shell "mkdir -p $WORK_SPACE/$DATASET_NAME/inputs"
      adb $ADB_DEVICE_NAME shell "rm -rf $WORK_SPACE/$DATASET_NAME/outputs"
      adb $ADB_DEVICE_NAME shell "mkdir -p $WORK_SPACE/$DATASET_NAME/outputs"
      rm -rf list.txt
    fi
    adb $ADB_DEVICE_NAME push ../assets/datasets/$DATASET_NAME/inputs/$SAMPLE_NAME $WORK_SPACE/$DATASET_NAME/inputs/
    echo -e "$SAMPLE_NAME" >> list.txt
    SPLIT_INDEX=$(($SPLIT_INDEX + 1))
  else
    echo "skip..."
  fi 
  SAMPLE_INDEX=$(($SAMPLE_INDEX + 1))
done
if [ $SPLIT_INDEX -gt 0 ] ; then
  adb $ADB_DEVICE_NAME push list.txt $WORK_SPACE/$DATASET_NAME/
  adb $ADB_DEVICE_NAME shell "$COMMAND_LINE"
  adb $ADB_DEVICE_NAME pull $WORK_SPACE/$DATASET_NAME/outputs/ ../assets/datasets/$DATASET_NAME/outputs/
fi
rm -rf list.txt
adb $ADB_DEVICE_NAME pull ${WORK_SPACE}/${MODEL_NAME}.nb ../assets/models/
if [ "$NNADAPTER_MODEL_CACHE_DIR" != "null" ]; then
  adb $ADB_DEVICE_NAME pull $WORK_SPACE/$NNADAPTER_MODEL_CACHE_DIR ../assets/models/
fi
