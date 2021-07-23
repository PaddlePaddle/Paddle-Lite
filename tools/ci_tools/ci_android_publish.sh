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
# Toolchain: gcc or clang, default gcc.
TOOLCHAIN=(gcc clang)
# WITH_EXTRA: with extra ops or not
WITH_EXTRA=(ON OFF)
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
#   3. with extra or not
# Returns:
#   None
####################################################################################################
function publish_inference_lib {
  local arch=$1
  local toolchain=$2
  local with_extra=$3
  cd $WORKSPACE
  # Remove Compiling Cache
  rm -rf build*
  # Compiling inference library
  ./lite/tools/build_android.sh --arch=$arch --toolchain=$toolchain  --with_extra=$with_extra  --with_static_lib=ON
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
  echo -e "*     ./lite/tools/build_android.sh --arch=$arch --toolchain=$toolchain  --with_extra=ON"
  echo "**************************************************************************************"
  exit 1
}


# Compiling test
for arch in ${ARCH[@]}; do
  for toolchain in ${TOOLCHAIN[@]}; do
    for with_extra in ${WITH_EXTRA[@]}; do
      cd $WORKSPACE
      publish_inference_lib $arch $toolchain $with_extra
    done
  done
done    
