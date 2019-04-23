#!/usr/bin/env bash

# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -e
source ./ci_run_test.sh

function print_usage() {
  echo "\n${RED}Usage${NONE}:
  ${BOLD}${SCRIPT_NAME}${NONE} [Option] [Network]"

  echo "\n${RED}Option${NONE}: required, specify the target platform
  ${BLUE}android_armv7${NONE}: run build for android armv7 platform
  ${BLUE}android_armv8${NONE}: run build for android armv8 platform
  ${BLUE}ios${NONE}: run build for apple ios platform
  ${BLUE}linux_armv7${NONE}: run build for linux armv7 platform
  ${BLUE}linux_armv8${NONE}: run build for linux armv8 platform
  ${BLUE}fpga${NONE}: run build for fpga platform
  "
  echo "\n${RED}Network${NONE}: optional, for deep compressing the framework size
  ${BLUE}googlenet${NONE}: build only googlenet support
  ${BLUE}mobilenet${NONE}: build only mobilenet support
  ${BLUE}yolo${NONE}: build only yolo support
  ${BLUE}squeezenet${NONE}: build only squeezenet support
  ${BLUE}resnet${NONE}: build only resnet support
  ${BLUE}mobilenetssd${NONE}: build only mobilenetssd support
  ${BLUE}nlp${NONE}: build only nlp model support
  ${BLUE}mobilenetfssd${NONE}: build only mobilenetfssd support
  ${BLUE}genet${NONE}: build only genet support
  ${BLUE}super${NONE}: build only super support
  "
}

function init() {
  RED='\033[0;31m'
  BLUE='\033[0;34m'
  BOLD='\033[1m'
  NONE='\033[0m'

  PADDLE_MOBILE_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}")/../" && pwd )"
  if [ -z "${SCRIPT_NAME}" ]; then
      SCRIPT_NAME=$0
  fi
}

function check_ndk() {
  if [ -z "${NDK_ROOT}" ]; then
    echo "Should set NDK_ROOT as your android ndk path, such as\n"
    echo "  export NDK_ROOT=~/android-ndk-r14b\n"
    exit -1
  fi
}

function build_android_armv7_cpu_only() {
#  rm -rf ../build/armeabi-v7a
  cmake .. \
    -B"../build/armeabi-v7a" \
    -DANDROID_ABI="armeabi-v7a with NEON" \
    -DCMAKE_BUILD_TYPE="MinSizeRel" \
    -DCMAKE_TOOLCHAIN_FILE="./tools/android-cmake/android.toolchain.cmake" \
    -DANDROID_PLATFORM="android-22" \
    -DANDROID_STL=c++_static \
    -DANDROID=true \
    -DWITH_LOGGING=OFF \
    -DCPU=ON \
    -DGPU_CL=OFF \
    -DFPGA=OFF

  cd ../build/armeabi-v7a && make -j 8
  cd -
}

function build_android_armv7_gpu() {
  rm -rf ../build/armeabi-v7a
  cmake .. \
    -B"../build/armeabi-v7a" \
    -DANDROID_ABI="armeabi-v7a with NEON" \
    -DCMAKE_BUILD_TYPE="MinSizeRel" \
    -DCMAKE_TOOLCHAIN_FILE="./tools/android-cmake/android.toolchain.cmake" \
    -DANDROID_PLATFORM="android-22" \
    -DANDROID_STL=c++_static \
    -DANDROID=true \
    -DWITH_LOGGING=OFF \
    -DCPU=ON \
    -DGPU_CL=ON \
    -DFPGA=OFF

  cd ../build/armeabi-v7a && make -j 8
  cd -
}

function build_android_armv8_cpu_only() {
  rm -rf ../build/arm64-v8a
  cmake .. \
    -B"../build/arm64-v8a" \
    -DANDROID_ABI="arm64-v8a" \
    -DCMAKE_BUILD_TYPE="MinSizeRel" \
    -DCMAKE_TOOLCHAIN_FILE="./tools/android-cmake/android.toolchain.cmake" \
    -DANDROID_PLATFORM="android-22" \
    -DANDROID_STL=c++_static \
    -DANDROID=true \
    -DWITH_LOGGING=OFF \
    -DCPU=ON \
    -DGPU_CL=OFF \
    -DFPGA=OFF

  cd ../build/arm64-v8a && make -j 1
  cd -
}

function build_android_armv8_gpu() {
  rm -rf ../build/arm64-v8a
  cmake .. \
    -B"../build/arm64-v8a" \
    -DANDROID_ABI="arm64-v8a" \
    -DCMAKE_BUILD_TYPE="MinSizeRel" \
    -DCMAKE_TOOLCHAIN_FILE="./tools/android-cmake/android.toolchain.cmake" \
    -DANDROID_PLATFORM="android-22" \
    -DANDROID_STL=c++_static \
    -DANDROID=true \
    -DWITH_LOGGING=OFF \
    -DCPU=ON \
    -DGPU_CL=ON \
    -DFPGA=OFF

  cd ../build/arm64-v8a && make -j 8
  cd -
}

function build_ios_armv8_cpu_only() {
  rm -rf ../build/ios
  cmake .. \
    -B"../build/ios" \
    -DCMAKE_BUILD_TYPE="MinSizeRel" \
    -DCMAKE_TOOLCHAIN_FILE="./tools/ios-cmake/ios.toolchain.cmake" \
    -DIOS_PLATFORM=OS \
    -DIOS_ARCH="${IOS_ARCH}" \
    -DIS_IOS=true \
    -DUSE_OPENMP=OFF \
    -DCPU=ON \
    -DGPU_CL=OFF \
    -DFPGA=OFF

  cd ../build/ios && make -j 8
  cd -
}

function build_ios_armv8_gpu() {
  rm -rf ../build/ios
  cmake .. \
    -B"../build/ios" \
    -DCMAKE_BUILD_TYPE="MinSizeRel" \
    -DCMAKE_TOOLCHAIN_FILE="./tools/ios-cmake/ios.toolchain.cmake" \
    -DIOS_PLATFORM=OS \
    -DIOS_ARCH="${IOS_ARCH}" \
    -DIS_IOS=true \
    -DUSE_OPENMP=OFF \
    -DCPU=ON \
    -DGPU_CL=ON \
    -DFPGA=OFF

  cd ../build/ios && make -j 8
  cd -
}

function build_linux_armv7_cpu_only() {
  rm -rf ../build/armv7_linux
  cmake .. \
    -B"../build/armv7_linux" \
    -DCMAKE_BUILD_TYPE="MinSizeRel" \
    -DCMAKE_TOOLCHAIN_FILE="./tools/toolchains/arm-linux-gnueabihf.cmake" \
    -DCPU=ON \
    -DGPU_CL=OFF \
    -DFPGA=OFF

  cd ../build/armv7_linux && make -j 8
  cd -
}

function build_linux_armv7_gpu() {
  rm -rf ../build/armv7_linux
  cmake .. \
    -B"../build/armv7_linux" \
    -DCMAKE_BUILD_TYPE="MinSizeRel" \
    -DCMAKE_TOOLCHAIN_FILE="./tools/toolchains/arm-linux-gnueabihf.cmake" \
    -DCPU=ON \
    -DGPU_CL=ON \
    -DFPGA=OFF

  cd ../build/armv7_linux && make -j 8
  cd -
}

function build_android_armv7() {
  check_ndk
  build_android_armv7_cpu_only
  # build_android_armv7_gpu
}

function build_android_armv8() {
  check_ndk
  build_android_armv8_cpu_only
  # build_android_armv8_gpu
}

function build_ios() {
  build_ios_armv8_cpu_only
  # build_ios_armv8_gpu
}

function build_linux_armv7() {
  build_linux_armv7_cpu_only
  # build_linux_armv7_gpu
}

function build_linux_fpga() {
  cd ..
  image=`docker images paddle-mobile:dev | grep 'paddle-mobile'`
  if [[ "x"$image == "x" ]]; then
    docker build -t paddle-mobile:dev - < Dockerfile
  fi
  docker run --rm -v `pwd`:/workspace paddle-mobile:dev bash /workspace/tools/docker_build_fpga.sh
  cd -
}

function run_android_test() {
  ExecuteAndroidTests $1
}

function main() {
  local CMD=$1
  init
  case $CMD in
    android_armv7)
      build_android_armv7
      run_android_test armeabi-v7a 
      ;;
    android_armv8)
      build_android_armv8
      run_android_test arm64-v8a
      ;;
    ios)
      build_ios
      ;;
    linux_armv7)
      build_linux_armv7
      ;;
    fpga)
      build_linux_fpga
      ;;
    *)
      print_usage
      exit 0
      ;;
    esac
}

main $@
