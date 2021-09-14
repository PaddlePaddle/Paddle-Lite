#!/bin/bash
#
# Start the CI task of examining Android inference lib compiling.
set +x
set -e

#####################################################################################################
# Usage: test the publish period on Android platform.
# Data: 20210104
# Author: DannyIsFunny
#####################################################################################################

#####################################################################################################
# 1. global variables, you can change them according to your requirements
#####################################################################################################
# Architecture: armv7 or armv8, default armv8.
ARCH=(armv8 armv7)
# Toolchain: clang or gcc, default clang.
TOOLCHAIN=(clang gcc)
# OpenCL
OPENCL=(ON OFF)
# Absolute path of Paddle-Lite source code.
SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
WORKSPACE=${SHELL_FOLDER%tools/ci_tools*}
#####################################################################################################

####################################################################################################
# Functions of Android compiling test.
# Globals:
#   WORKSPACE
# Arguments:
#   1. architecture
#   2. toolchain
#   3. support opencl or not
#   4. with extra or not
# Returns:
#   None
####################################################################################################
function publish_inference_lib {
  local arch=$1
  local toolchain=$2
  local with_opencl=$3
  local with_extra=$4
  cd $WORKSPACE
  # Remove Compiling Cache
  rm -rf build*
  # Compiling inference library
  cmd="./lite/tools/build_android.sh --arch=$arch --toolchain=$toolchain --with_extra=$with_extra --with_opencl=$with_opencl --with_static_lib=ON"
  ${cmd}
  # Checking results: cplus and java inference lib.
  if [ -d build*/inference*/cxx/lib ] && [ -d build*/inference*/java/so ]; then
    cxx_results=$(ls build*/inference*/cxx/lib | wc -l)
    java_results=$(ls build*/inference*/java/so | wc -l)
      if [ $cxx_results -ge 2 ] && [ $java_results -ge 1 ]; then
          return 0
      fi
  fi
  # Error message.
  echo "**************************************************************************************"
  echo -e "* Android compiling task failed on the following instruction:"
  echo -e "*    ${cmd}"
  echo "**************************************************************************************"
  exit 1
}


# Compiling test: Not fully test for time-saving
for arch in armv8; do
  for toolchain in clang; do
    for opencl in ${OPENCL[@]}; do
      cd $WORKSPACE
      publish_inference_lib $arch $toolchain $opencl ON
    done
  done
done
