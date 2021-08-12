#!/bin/bash
#
# Start the CI task of examining iOS metal-based inference lib compiling.
set +x
set -e

#####################################################################################################
# Usage: test the publish period on iOS metal-based platform.
# Data: 20210125
# Author: DannyIsFunny
#####################################################################################################

#####################################################################################################
# 1. global variables, you can change them according to your requirements
#####################################################################################################
# Architecture: armv7 or armv8, default armv8.
ARCH=(armv8 armv7)
# Absolute path of Paddle-Lite source code.
SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
WORKSPACE=${SHELL_FOLDER%tools/ci_tools*}
#####################################################################################################

####################################################################################################
# Functions of iOS compiling test.
# Globals:
#   WORKSPACE
# Arguments:
#   1. architecture
#   2. with extra or not
# Returns:
#   None
####################################################################################################
function publish_inference_lib {
  local arch=$1
  local with_extra=$2
  cd $WORKSPACE
  # Remove Compiling Cache
  rm -rf build*
  # Compiling inference library
  ./lite/tools/build_ios.sh --with_metal=ON --arch=$arch --with_extra=$with_extra
  # Checking results: cplus and java inference lib.
  if [ -d build*/inference*/lib ]; then
    cxx_results=$(ls build*/inference*/lib | wc -l)
    if [ $cxx_results -ge 1 ] ; then
        return 0
    fi
  fi
  # Error message.
  echo "**************************************************************************************"
  echo -e "* iOS metal-based compiling task failed on the following instruction:"
  echo -e "*     ./lite/tools/build_ios.sh --arch=$arch --with_metal=ON --with_extra=ON"
  echo "**************************************************************************************"
  exit 1
}


# Compiling test
for arch in ${ARCH[@]}; do
  cd $WORKSPACE
  publish_inference_lib $arch ON
done    
