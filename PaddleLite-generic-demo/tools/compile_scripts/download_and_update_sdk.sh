#!/bin/bash
set -e

source settings.sh

if [ "$ENABLE_BUILD_HUAWEI_KIRIN_NPU" == "1" ]; then
  curl -L http://paddlelite-demo.bj.bcebos.com/devices/huawei/kirin/hiai_ddk_lib_510.tar.gz -o - | tar -xz -C $SDK_DIR
fi

if [ "$ENABLE_BUILD_MEDIATEK_APU" == "1" ]; then
  curl -L http://paddlelite-demo.bj.bcebos.com/devices/mediatek/apu_ddk.tar.gz -o - | tar -xz -C $SDK_DIR
fi

if [ "$ENABLE_BUILD_IMAGINATION_NNA" == "1" ]; then
  curl -L http://paddlelite-demo.bj.bcebos.com/devices/imagination/imagination_nna_sdk.tar.gz -o - | tar -xz -C $SDK_DIR
fi

if [ "$ENABLE_BUILD_HUAWEI_ASCEND_NPU" == "1" ]; then
  curl -L http://paddlelite-demo.bj.bcebos.com/devices/huawei/ascend/kunpeng920_arm/ascend-toolkit-aarch64-linux_3.3.0.tar.gz -o - | tar -xz -C $SDK_DIR
  curl -L http://paddlelite-demo.bj.bcebos.com/devices/huawei/ascend/kunpeng920_arm/ascend-toolkit-aarch64-linux_5.1.RC1.alpha001.tar.gz -o - | tar -xz -C $SDK_DIR
  curl -L http://paddlelite-demo.bj.bcebos.com/devices/huawei/ascend/intel_x86/ascend-toolkit-x86_64-linux_3.3.0.tar.gz -o - | tar -xz -C $SDK_DIR
  curl -L http://paddlelite-demo.bj.bcebos.com/devices/huawei/ascend/intel_x86/ascend-toolkit-x86_64-linux_5.1.RC1.alpha001.tar.gz -o - | tar -xz -C $SDK_DIR
fi

if [ "$ENABLE_BUILD_CAMBRICON_MLU" == "1" ]; then
  curl -L http://paddlelite-demo.bj.bcebos.com/devices/cambricon_mlu/neuware_2.8.3.tar.gz -o - | tar -xz -C $SDK_DIR
fi

if [ "$ENABLE_BUILD_INTEL_OPENVINO" == "1" ]; then
  curl -L http://paddlelite-demo.bj.bcebos.com/devices/intel/openvino_2022.1.0.643.tar.gz -o - | tar -xz -C $SDK_DIR
fi

echo "Done."
