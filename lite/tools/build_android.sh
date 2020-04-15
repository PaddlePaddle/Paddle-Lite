#!/bin/bash
set -ex

#####################################################################################################
# 1. global variables, you can change them according to your requirements
#####################################################################################################
# armv7 or armv8, default armv8.
ARM_ABI=armv8
# c++_static or c++_shared, default c++_static.
ANDROID_STL=c++_static
# gcc or clang, default gcc.
ARM_LANG=gcc
# ON or OFF, default OFF.
BUILD_EXTRA=OFF
# ON or OFF, default ON. 
BUILD_JAVA=ON
# controls whether to compile cv functions into lib, default is OFF.
BUILD_CV=OFF
# controls whether to hide log information, default is ON.
SHUTDOWN_LOG=ON
BUILD_DIR=$(pwd)
# options of striping lib according to input model.
OPTMODEL_DIR=""
BUILD_TAILOR=OFF
# num of threads used during compiling..
readonly NUM_PROC=${LITE_BUILD_THREADS:-4}
#####################################################################################################




#####################################################################################################
# 2. local variables, these variables should not be changed.
#####################################################################################################
# url that stores third-party zip file to accelerate third-paty lib installation
readonly THIRDPARTY_TAR=https://paddle-inference-dist.bj.bcebos.com/PaddleLite/third-party-05b862.tar.gz
readonly workspace=$PWD
# basic options for android compiling.
readonly CMAKE_COMMON_OPTIONS="-DWITH_LITE=ON \
                               -DLITE_WITH_ARM=ON \
                               -DLITE_WITH_LIGHT_WEIGHT_FRAMEWORK=ON"
# on mac environment, we should expand the maximum file num to compile successfully
os_name=`uname -s`
if [ ${os_name} == "Darwin" ]; then
   ulimit -n 1024
fi
#####################################################################################################





####################################################################################################
# 3. functions of prepare workspace before compiling
####################################################################################################

# 3.1 generate `__generated_code__.cc`, which is dependended by some targets in cmake.
# here we fake an empty file to make cmake works.
function prepare_workspace {
    local root_dir=$1
    local build_dir=$2
    # ARM LANG
    if [ ${ARM_LANG} == "clang" ]; then
        LITE_WITH_ARM_LANG=ON
    else
        LITE_WITH_ARM_LANG=OFF
    fi
    echo "ARM_LANG is  ${ARM_LANG}"
    echo "LITE_WITH_ARM_LANG is ${LITE_WITH_ARM_LANG}"
    # in build directory
    # 1. Prepare gen_code file
    GEN_CODE_PATH_PREFIX=$build_dir/lite/gen_code
    mkdir -p ${GEN_CODE_PATH_PREFIX}
    touch ${GEN_CODE_PATH_PREFIX}/__generated_code__.cc
    # 2.Prepare debug tool
    DEBUG_TOOL_PATH_PREFIX=$build_dir/lite/tools/debug
    mkdir -p ${DEBUG_TOOL_PATH_PREFIX}
    cp $root_dir/lite/tools/debug/analysis_tool.py ${DEBUG_TOOL_PATH_PREFIX}/
}


# 3.2 prepare source code of opencl lib
# here we bundle all cl files into a cc file to bundle all opencl kernels into a single lib
function prepare_opencl_source_code {
    local root_dir=$1
    local build_dir=$2
    # in build directory
    # Prepare opencl_kernels_source.cc file
    GEN_CODE_PATH_OPENCL=$root_dir/lite/backends/opencl
    rm -f GEN_CODE_PATH_OPENCL/opencl_kernels_source.cc
    OPENCL_KERNELS_PATH=$root_dir/lite/backends/opencl/cl_kernel
    mkdir -p ${GEN_CODE_PATH_OPENCL}
    touch $GEN_CODE_PATH_OPENCL/opencl_kernels_source.cc
    python $root_dir/lite/tools/cmake_tools/gen_opencl_code.py $OPENCL_KERNELS_PATH $GEN_CODE_PATH_OPENCL/opencl_kernels_source.cc 
}

# 3.3 prepare third_party libraries for compiling
# here we store third_party libraries into Paddle-Lite/third-party
function prepare_thirdparty {
    if [ ! -d $workspace/third-party -o -f $workspace/third-party-05b862.tar.gz ]; then
        rm -rf $workspace/third-party

        if [ ! -f $workspace/third-party-05b862.tar.gz ]; then
            wget $THIRDPARTY_TAR
        fi
        tar xzf third-party-05b862.tar.gz
    else
        git submodule update --init --recursive
    fi
}
####################################################################################################





####################################################################################################
# 4. compiling functions
####################################################################################################

# 4.1 function of tiny_publish compiling
# here we only compile light_api lib
function make_tiny_publish_so {
#  local os=$1
  local abi=$1
  local lang=$2
  local android_stl=$3

  cur_dir=$(pwd)
  build_dir=$cur_dir/build.lite.adnroid.${abi}.${lang}
  if [ -d $build_dir ]
  then
    rm -rf $build_dir
  fi
  mkdir -p $build_dir
  cd $build_dir

  cmake .. \
      ${PYTHON_FLAGS} \
      ${CMAKE_COMMON_OPTIONS} \
      -DWITH_TESTING=OFF \
      -DLITE_WITH_JAVA=$BUILD_JAVA \
      -DLITE_WITH_PYTHON=$BUILD_PYTHON \
      -DLITE_SHUTDOWN_LOG=$SHUTDOWN_LOG \
      -DLITE_ON_TINY_PUBLISH=ON \
      -DANDROID_STL_TYPE=$android_stl \
      -DLITE_BUILD_EXTRA=$BUILD_EXTRA \
      -DLITE_WITH_CV=$BUILD_CV \
      -DLITE_WITH_ARM_LANG=$LITE_WITH_ARM_LANG \
      -DLITE_BUILD_TAILOR=$BUILD_TAILOR \
      -DLITE_OPTMODEL_DIR=$OPTMODEL_DIR \
      -DLITE_WITH_NPU=$BUILD_NPU \
      -DLITE_WITH_X86=OFF \
      -DNPU_DDK_ROOT=$NPU_DDK_ROOT \
      -DLITE_WITH_XPU=$BUILD_XPU \
      -DLITE_WITH_XTCL=$BUILD_XTCL \
      -DXPU_SDK_ROOT=$XPU_SDK_ROOT \
      -DARM_TARGET_OS=android -DARM_TARGET_ARCH_ABI=${abi} -DARM_TARGET_LANG=${lang}

  make publish_inference -j$NUM_PROC
  cd - > /dev/null
}

# 4.2 function of full_publish compiling
# here we compile both light_api lib and full_api lib
function make_full_publish_so {
  local abi=$1
  local lang=$2
  local android_stl=$3

  prepare_thirdparty

  root_dir=$(pwd)
  build_directory=$BUILD_DIR/build.lite.android.${abi}.${lang}

  if [ -d $build_directory ]
  then
    rm -rf $build_directory
  fi
  mkdir -p $build_directory
  cd $build_directory

  prepare_workspace $root_dir $build_directory
  cmake $root_dir \
      ${PYTHON_FLAGS} \
      ${CMAKE_COMMON_OPTIONS} \
      -DWITH_TESTING=OFF \
      -DLITE_WITH_JAVA=$BUILD_JAVA \
      -DLITE_WITH_PYTHON=$BUILD_PYTHON \
      -DLITE_SHUTDOWN_LOG=$SHUTDOWN_LOG \
      -DANDROID_STL_TYPE=$android_stl \
      -DLITE_BUILD_EXTRA=$BUILD_EXTRA \
      -DLITE_WITH_X86=OFF \
      -DLITE_WITH_CV=$BUILD_CV \
      -DLITE_WITH_ARM_LANG=$LITE_WITH_ARM_LANG \
      -DLITE_BUILD_TAILOR=$BUILD_TAILOR \
      -DLITE_OPTMODEL_DIR=$OPTMODEL_DIR \
      -DLITE_WITH_NPU=$BUILD_NPU \
      -DNPU_DDK_ROOT=$NPU_DDK_ROOT \
      -DLITE_WITH_XPU=$BUILD_XPU \
      -DLITE_WITH_XTCL=$BUILD_XTCL \
      -DXPU_SDK_ROOT=$XPU_SDK_ROOT \
      -DLITE_WITH_TRAIN=$BUILD_TRAIN \
      -DARM_TARGET_OS=android -DARM_TARGET_ARCH_ABI=${abi} -DARM_TARGET_LANG=${lang}

  make publish_inference -j$NUM_PROC
  cd - > /dev/null
}

# 4.3 function of opencl compiling
# here we compile both light_api and full_api opencl lib
function make_opencl {
  local abi=$1
  local lang=$2
  prepare_thirdparty

  root_dir=$(pwd)
  build_dir=$root_dir/build.lite.android.${abi}.${lang}.opencl
  if [ -d $build_directory ]
  then
  rm -rf $build_directory
  fi
  mkdir -p $build_dir
  cd $build_dir
  prepare_workspace $root_dir $build_dir
  prepare_opencl_source_code $root_dir $build_dir
  cmake .. \
      -DLITE_WITH_OPENCL=ON \
      -DWITH_GPU=OFF \
      -DWITH_MKL=OFF \
      -DWITH_LITE=ON \
      -DLITE_WITH_CUDA=OFF \
      -DLITE_WITH_X86=OFF \
      -DLITE_WITH_ARM=ON \
      -DWITH_ARM_DOTPROD=ON   \
      -DLITE_ON_TINY_PUBLISH=ON \
      -DLITE_WITH_LIGHT_WEIGHT_FRAMEWORK=ON \
      -DWITH_TESTING=OFF \
      -DLITE_BUILD_EXTRA=$BUILD_EXTRA \
      -DLITE_SHUTDOWN_LOG=$SHUTDOWN_LOG \
      -DLITE_WITH_CV=$BUILD_CV \
      -DARM_TARGET_OS=android -DARM_TARGET_ARCH_ABI=$abi -DARM_TARGET_LANG=$lang

    make opencl_clhpp -j$NUM_PROC
    make publish_inference -j$NUM_PROC
}
####################################################################################################



function print_usage {
    set +x
    echo -e "\n Methods of compiling Padddle-Lite android library:"
    echo "----------------------------------------"
    echo -e "compile light_api lib (recommanded): (armv8, gcc, c++_static)"
    echo -e "   ./build.sh tiny_publish"
    echo -e "compile both light_api and cxx_api lib: (armv8, gcc, c++_static)"
    echo -e "   ./build.sh full_publish"
    echo
    echo -e "optional argument:"
    echo -e "--arm_abi:\t armv8|armv7, default is armv8"
    echo -e "--arm_lang:\t gcc|clang, defalut is gcc"
    echo -e "--android_stl:\t c++_static|c++_shared, default is c++_static"
    echo -e "--shutdown_log: (OFF|ON); controls whether to shutdown log, default is ON"
    echo -e "--build_python: (OFF|ON); controls whether to publish python api lib, default is OFF"
    echo -e "--build_java: (OFF|ON); controls whether to publish java api lib, default is ON"
    echo -e "--build_cv: (OFF|ON); controls whether to compile cv functions into lib, default is OFF"
    echo -e "--shutdown_log: (OFF|ON); controls whether to hide log information, default is ON"
    echo -e "--build_extra: (OFF|ON); controls whether to publish extra operators and kernels for (sequence-related model such as OCR or NLP)"
    echo
    echo -e "arguments of striping lib according to input model:"
    echo -e "--build_strip: (OFF|ON); controls whether to strip lib accrding to input model, default is OFF"
    echo -e "--opt_model_dir: (path to optimized model dir); contains absolute path to optimized model dir"
    echo "----------------------------------------"
    echo
}

function main {
    if [ -z "$1" ]; then
        print_usage
        exit -1
    fi

    # Parse command line.
    for i in "$@"; do
        case $i in
            # armv7 or armv8, default armv8
            --arm_abi=*)
                ARM_ABI="${i#*=}"
                shift
                ;;
            # gcc or clang, default gcc
            --arm_lang=*)
                ARM_LANG="${i#*=}"
                shift
                ;;
            # c++_static or c++_shared, default c++_static
            --android_stl=*)
                ANDROID_STL="${i#*=}"
                shift
                ;;
            # ON or OFF, default OFF
            --build_extra=*)
                BUILD_EXTRA="${i#*=}"
                shift
                ;;
            # ON or OFF, default OFF
            --build_cv=*)
                BUILD_CV="${i#*=}"
                shift
                ;;
            # ON or OFF, default ON
            --build_java=*)
                BUILD_JAVA="${i#*=}"
                shift
                ;;
            # string, absolute path to optimized model dir
            --opt_model_dir=*)
                OPTMODEL_DIR="${i#*=}"
                shift
                ;;
            # ON or OFF, default OFF
            --build_tailor=*)
                BUILD_TAILOR="${i#*=}"
                shift
                ;;
            # ON or OFF, default ON
            --shutdown_log=*)
                SHUTDOWN_LOG="${i#*=}"
                shift
                ;;
            # compiling result contains light_api lib only, recommanded.
            tiny_publish)
                make_tiny_publish_so $ARM_ABI $ARM_LANG $ANDROID_STL 
                shift
                ;;
            # compiling result contains both light_api and cxx_api lib.
            full_publish)
                make_full_publish_so $ARM_ABI $ARM_LANG $ANDROID_STL 
                shift
                ;;
            # compiling lib which can operate on opencl and cpu.
            opencl)
                make_opencl $ARM_ABI $ARM_LANG
                shift
                ;;
            *)
                # unknown option
                print_usage
                exit 1
                ;;
        esac
    done
}

main $@
