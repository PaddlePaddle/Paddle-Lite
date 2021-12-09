#!/bin/bash
#
# Start the CI task of examining iOS inference lib compiling.
set +x
set -e

#####################################################################################################
# Usage: test the publish period on iOS platform.
# Data: 20210125
# Author: DannyIsFunny
#####################################################################################################

#####################################################################################################
# 1. global variables, you can change them according to your requirements
#####################################################################################################
# Architecture: armv8, default armv8.
ARCH=(armv8)
# Whether to use GPU or not: ON or OFF, default OFF.
USE_GPU=(ON OFF)
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
#   2. whether to use GPU or not
#   3. with extra or not
# Returns:
#   None
####################################################################################################
function publish_inference_lib {
  local arch=$1
  local with_metal=$2
  local with_extra=$3
  cd $WORKSPACE
  # Remove Compiling Cache
  rm -rf build*
  # Compiling inference library
  ./lite/tools/build_ios.sh --arch=$arch --with_metal=${with_metal} --with_extra=${with_extra}
  # Checking results: cplus and java inference lib.
  if [ -d build*/inference*/lib ]; then
    cxx_results=$(ls build*/inference*/lib | wc -l)
    if [ $cxx_results -ge 1 ] ; then
        return 0
    fi
  fi
  # Error message.
  echo "**************************************************************************************"
  echo -e "* iOS compiling task failed on the following instruction:"
  echo -e "*     ./lite/tools/build_ios.sh --arch=$arch --with_metal=$with_metal --with_extra=ON"
  echo "**************************************************************************************"
  exit 1
}


# Compiling test
for arch in ${ARCH[@]}; do
  for use_gpu in ${USE_GPU[@]}; do
    cd $WORKSPACE
    publish_inference_lib $arch use_gpu ON
  done
done    
