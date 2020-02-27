#!/bin/bash
set -ex

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
LITE_DIR=$(readlink -f "${SCRIPT_DIR}/../../")
cd "$LITE_DIR"

pushd ../
if [ ! -d XPU_SDK_gcc82 ]; then
  wget --http-user=isa --http-passwd=isa#123 http://bb-inf-sat6.bb01:8080/XPU_SDK_gcc82.tar.gz
  tar -zxvf XPU_SDK_gcc82.tar.gz
fi
popd

git submodule init
git submodule update --recursive

mkdir -p build.lite.xpu.gcc82
pushd build.lite.xpu.gcc82

env CC=/opt/compiler/gcc-8.2/bin/gcc CXX=/opt/compiler/gcc-8.2/bin/g++ \
  cmake .. -G "Eclipse CDT4 - Unix Makefiles" \
  -DCMAKE_BUILD_TYPE=Debug \
  -DWITH_LITE=ON \
  -DLITE_WITH_LIGHT_WEIGHT_FRAMEWORK=OFF \
  -DWITH_PYTHON=OFF \
  -DLITE_WITH_ARM=OFF \
  -DWITH_GPU=OFF \
  -DWITH_MKLDNN=OFF \
  -DLITE_WITH_X86=ON \
  -DWITH_MKL=ON \
  -DLITE_WITH_XPU=ON \
  -DLITE_BUILD_EXTRA=ON \
  -DWITH_TESTING=ON \
  -DXPU_SDK_ROOT=$(readlink -f ../../XPU_SDK_gcc82)
make -j32 test_bert_lite_xpu
