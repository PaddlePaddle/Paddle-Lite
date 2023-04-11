
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
ADNN_ARCH=armv8
# Set the type of ADNN library: shared, static, or default, defaults to default.
ADNN_LIBRARY_TYPE=default
# Set the type of target: Debug, Release, RelWithDebInfo and MinSizeRel, defaults to Release.
CMAKE_BUILD_TYPE=Release
# Throw an exception when error occurs, defaults to OFF.
ADNN_WITH_EXCEPTION=OFF
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
  cmake_args+=("-DADNN_LIBRARY_TYPE=$ADNN_LIBRARY_TYPE")
  if [ "$ADNN_ARCH" == "armv8" ]; then
    cmake_args+=("-DCMAKE_TOOLCHAIN_FILE=cmake/platforms/linux/aarch64.toolchain.cmake")
  elif [ "$ADNN_ARCH" == "armv7hf" ]; then
    cmake_args+=("-DCMAKE_TOOLCHAIN_FILE=cmake/platforms/linux/armhf.toolchain.cmake")
  else
    echo "Unsupported arch $ADNN_ARCH."
    exit 1
  fi

  build_dir=build/linux/$ADNN_ARCH
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
        ADNN_ARCH="${i#*=}"
        shift
        ;;
      --with_exception=*)
        ADNN_WITH_EXCEPTION="${i#*=}"
        shift
        ;;
      --library_type=*)
        ADNN_LIBRARY_TYPE="${i#*=}"
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
