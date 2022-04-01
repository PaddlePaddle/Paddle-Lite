#!/bin/bash
set -e

# Settings only for Android
ANDROID_NDK=/opt/android-ndk-r17c # docker
#ANDROID_NDK=/Users/hongming/Library/android-ndk-r17c # macOS

# For TARGET_OS=android, TARGET_ABI=arm64-v8a/armeabi-v7a.
# For TARGET_OS=linux, TARGET_ABI=arm64/armhf/amd64.
TARGET_OS=linux
if [ -n "$1" ]; then
  TARGET_OS=$1
fi

TARGET_ABI=arm64
if [ -n "$2" ]; then
  TARGET_ABI=$2
fi

function readlinkf() {
  perl -MCwd -e 'print Cwd::abs_path shift' "$1";
}

# Set the corresponding libnnadapter.so path and include director for the current TARGET_OS and TARGET_ABI.
NNADAPTER_RUNTIME_INCLUDE_DIRECTORY=$(readlinkf ../../$TARGET_OS/$TARGET_ABI/lib/builtin_device/include)
NNADAPTER_RUNTIME_LIBRARY_PATH=$(readlinkf ../../${TARGET_OS}/$TARGET_ABI/lib/builtin_device/libnnadapter.so)
# Set the output directory of the NNAdapter driver HAL library.
NNADAPTER_DRIVER_LIBRARY_DIRECTORY=$(readlinkf ../../$TARGET_OS/$TARGET_ABI/lib/fake_device)

# Initialize the cmake compilation environment for the current TARGET_OS and TARGET_ABI.
CMAKE_COMMAND_ARGS="-DCMAKE_VERBOSE_MAKEFILE=ON -DNNADAPTER_WITH_STANDALONE=ON -DNNADAPTER_STANDALONE_RUNTIME_INCLUDE_DIRECTORY=$NNADAPTER_RUNTIME_INCLUDE_DIRECTORY -DNNADAPTER_STANDALONE_RUNTIME_LIBRARY_PATH=$NNADAPTER_RUNTIME_LIBRARY_PATH"
if [[ $TARGET_OS == "linux" ]]; then
  CMAKE_COMMAND_ARGS="$CMAKE_COMMAND_ARGS -DCMAKE_SYSTEM_NAME=Linux"
  if [[ $TARGET_ABI == "arm64" ]]; then
    CMAKE_COMMAND_ARGS="$CMAKE_COMMAND_ARGS -DCMAKE_SYSTEM_PROCESSOR=aarch64 -DCMAKE_C_COMPILER=aarch64-linux-gnu-gcc -DCMAKE_CXX_COMPILER=aarch64-linux-gnu-g++"
  elif [[ $TARGET_ABI == "armhf" ]]; then
    CMAKE_COMMAND_ARGS="$CMAKE_COMMAND_ARGS -DCMAKE_SYSTEM_PROCESSOR=arm -DCMAKE_C_COMPILER=arm-linux-gnueabihf-gcc -DCMAKE_CXX_COMPILER=arm-linux-gnueabihf-g++"
  elif [[ $TARGET_ABI == "amd64" ]]; then
    CMAKE_COMMAND_ARGS="$CMAKE_COMMAND_ARGS -DCMAKE_SYSTEM_PROCESSOR=x86_64 -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++"
  else
    echo "'linux' only supports 'arm64', 'armhf' and 'amd64'."
    exit -1
  fi
elif [[ $TARGET_OS == "android" ]]; then
  if [[ $TARGET_ABI == "arm64-v8a" ]]; then
    ANDROID_NATIVE_API_LEVEL=android-23
  elif [[ $TARGET_ABI == "armeabi-v7a" ]]; then
    ANDROID_NATIVE_API_LEVEL=android-21
  else
    echo "'android' only supports 'arm64-v8a' and 'armeabi-v7a'."
    exit -1
  fi
  CMAKE_COMMAND_ARGS="$CMAKE_COMMAND_ARGS -DCMAKE_TOOLCHAIN_FILE=${ANDROID_NDK}/build/cmake/android.toolchain.cmake -DANDROID_NDK=${ANDROID_NDK} -DANDROID_NATIVE_API_LEVEL=${ANDROID_NATIVE_API_LEVEL} -DANDROID_STL=c++_shared -DANDROID_ABI=${TARGET_ABI} -DANDROID_ARM_NEON=TRUE"
else
  echo "Unknown $TARGET_OS, only supports 'linux' and 'android'."
fi

# Create a temporary build directory, build and copy the generated dynamic libraries to the target driver HAL directory.
BUILD_DIR=build.$TARGET_OS.$TARGET_ABI

rm -rf $BUILD_DIR
mkdir $BUILD_DIR
cd $BUILD_DIR
cmake $CMAKE_COMMAND_ARGS ..
make

cp -rf *.so $NNADAPTER_DRIVER_LIBRARY_DIRECTORY
