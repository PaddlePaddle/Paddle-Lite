#! /bin/bash
set +x

#####################################################################################################
# Usage: test the publish period on Android platform.
# Data: 20210104
# Author: DannyIsFunny
#####################################################################################################

#####################################################################################################
# 1. global variables, you can change them according to your requirements
#####################################################################################################
# armv7 or armv8, default armv8.
ARCH=(armv8 armv7)
# gcc or clang, default gcc.
TOOLCHAIN=(gcc clang)
# absolute path of Paddle-Lite source code.
SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
WORKSPACE=${SHELL_FOLDER%tools/ci_tools*}
#####################################################################################################

####################################################################################################
# Functions of compiling test.
####################################################################################################
function publish_inference_lib {
    local arch=$1
    local toolchain=$2
    local with_extra=$3
    cd $WORKSPACE
    # Remove Compiling Cache
    rm -rf build*
    # Compiling inference library
    ./lite/tools/build_android.sh --arch=$arch --toolchain=$toolchain  --with_extra=$with_extra
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
    echo -e "*\033[1;31m Android compiling task failed on the following instruction: \033[0m"
    echo -e "*     \033[1;34m ./lite/tools/build_android.sh --arch=$arch --toolchain=$toolchain  --with_extra=ON \033[0m"
    echo "**************************************************************************************"
   
    echo "Error: Compiling Failed."
    exit 1
}


# Compiling test
for arch in ${ARCH[@]}
do
    for toolchain in ${TOOLCHAIN[@]}
    do
        cd $WORKSPACE
        publish_inference_lib $arch $toolchain ON
    done
done    
              
