#!/bin/bash
# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
# http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -e
set +x

# Configurable options
# armv7 or armv8, default armv8.
ARM_DNN_LIBRARY_ARCH=armv8
# gcc or clang, default gcc.
ARM_DNN_LIBRARY_TOOLCHAIN=gcc
# Set the type of ARM_DNN_LIBRARY library: shared, static, or default, defaults to default.
ARM_DNN_LIBRARY_LIBRARY_TYPE=default
# Set the type of target: Debug, Release, RelWithDebInfo and MinSizeRel, defaults to Release.
CMAKE_BUILD_TYPE=Release
# Path to Android NDK
ANDROID_NDK=$ANDROID_NDK
# Set the type of Android STL: c++_static or c++_shared, defaults to c++_static.
ANDROID_STL=c++_static
# Set the target android native api level: 16, 21, ... etc. defaults to default.
ANDROID_NATIVE_API_LEVEL="default"
# Set the default mininum android native api level.
MIN_ANDROID_NATIVE_API_LEVEL_ARMV7=16
MIN_ANDROID_NATIVE_API_LEVEL_ARMV8=21
# Throw an exception when error occurs, defaults to OFF.
ARM_DNN_LIBRARY_WITH_EXCEPTION=OFF
# Set the num of threads to build.
readonly NUM_PROC=${NUM_PROC:-4}

# On mac environment, we should expand the maximum file num to compile successfully.
os_name=`uname -s`
if [ $os_name == "Darwin" ]; then
  ulimit -n 1024
fi

function build {
  cmake_args=()
  cmake_args+=("-DCMAKE_BUILD_TYPE=$CMAKE_BUILD_TYPE")
  cmake_args+=("-DCMAKE_POSITION_INDEPENDENT_CODE=ON")
  cmake_args+=("-DARM_DNN_LIBRARY_LIBRARY_TYPE=$ARM_DNN_LIBRARY_LIBRARY_TYPE")

  # Android NDK toolchain depends on android ndk version.
  if [ "$ANDROID_NDK" ]; then
    name=$(echo $ANDROID_NDK | egrep -o "android-ndk-r[0-9]{2}")
    version=$(echo $name | egrep -o "[0-9]{2}")
    if [ "$version" -gt 17 ]; then
      ARM_DNN_LIBRARY_TOOLCHAIN=clang
    fi
  else
    echo "ANDROID_NDK not set."
    exit 1
  fi
  cmake_args+=("-DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake")

  # Android native api level depends on arch.
  if [ "$ARM_DNN_LIBRARY_ARCH" == "armv8" ]; then
    ANDROID_ABI=arm64-v8a
    MIN_ANDROID_NATIVE_API_LEVEL=$MIN_ANDROID_NATIVE_API_LEVEL_ARMV8
  elif [ "$ARM_DNN_LIBRARY_ARCH" == "armv7" ]; then
    ANDROID_ABI=armeabi-v7a
    MIN_ANDROID_NATIVE_API_LEVEL=$MIN_ANDROID_NATIVE_API_LEVEL_ARMV7
  else
    echo "Unsupported arch $ARM_DNN_LIBRARY_ARCH."
    exit 1
  fi
  cmake_args+=("-DANDROID_ABI=$ANDROID_ABI")
  if [ "$ANDROID_NATIVE_API_LEVEL" == "default" ]; then
    :
  elif [ $ANDROID_NATIVE_API_LEVEL -ge $MIN_ANDROID_NATIVE_API_LEVEL ]; then
    cmake_args+=("-DANDROID_NATIVE_API_LEVEL=$ANDROID_NATIVE_API_LEVEL")
  else
    echo "ANDROID_NATIVE_API_LEVEL should be no less than $MIN_ANDROID_NATIVE_API_LEVEL on $ARM_DNN_LIBRARY_ARCH."
    exit 1
  fi

  build_dir=build/android/$ARM_DNN_LIBRARY_ARCH
  if [ -d $build_dir ]; then
    rm -rf $build_dir
  fi
  mkdir -p $build_dir
  cd $build_dir && cmake ../../.. "${cmake_args[@]}"
  cmake --build . -- "-j$NUM_PROC"
  cd - > /dev/null
}

function main {
  for i in "$@"; do
    case $i in
      --arch=*)
        ARM_DNN_LIBRARY_ARCH="${i#*=}"
        shift
        ;;
      --toolchain=*)
        ARM_DNN_LIBRARY_TOOLCHAIN="${i#*=}"
        shift
        ;;
      --with_exception=*)
        ARM_DNN_LIBRARY_WITH_EXCEPTION="${i#*=}"
        if [[ $ARM_DNN_LIBRARY_WITH_EXCEPTION == "ON" && $ARM_DNN_LIBRARY_ARCH == "armv7" && $ARM_DNN_LIBRARY_TOOLCHAIN != "clang" ]]; then
          set +x
          echo
          echo -e "Only clang provide C++ exception handling support for 32-bit ARM."
          echo
          exit 1
        fi
        shift
        ;;
      --android_stl=*)
        ANDROID_STL="${i#*=}"
        shift
        ;;
      --android_native_api_level=*)
        ANDROID_NATIVE_API_LEVEL="${i#*=}"
        shift
        ;;
      --android_ndk=*)
        ANDROID_NDK="${i#*=}"
        shift
        ;;
      --library_type=*)
        ARM_DNN_LIBRARY_LIBRARY_TYPE="${i#*=}"
        shift
        ;;
      --build_type=*)
        CMAKE_BUILD_TYPE="${i#*=}"
        shift
        ;;
      *)
        echo "Unsupported argument \"${i#*=}\"."
        exit 1
        ;;
    esac
  done
  build
}

main $@
