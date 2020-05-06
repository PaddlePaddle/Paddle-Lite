#!/bin/bash
set -ex

readonly CMAKE_COMMON_OPTIONS="-DWITH_GPU=OFF \
                               -DWITH_MKL=OFF \
                               -DWITH_LITE=ON \
                               -DLITE_WITH_CUDA=OFF \
                               -DLITE_WITH_X86=OFF \
                               -DLITE_WITH_ARM=ON \
                               -DLITE_WITH_LIGHT_WEIGHT_FRAMEWORK=ON"

readonly NUM_PROC=${LITE_BUILD_THREADS:-4}


# global variables
BUILD_EXTRA=OFF
BUILD_TRAIN=OFF
BUILD_JAVA=ON
BUILD_PYTHON=OFF
BUILD_DIR=$(pwd)
OPTMODEL_DIR=""
BUILD_TAILOR=OFF
BUILD_CV=OFF
WITH_LOG=ON
BUILD_NPU=OFF
NPU_DDK_ROOT="$(pwd)/ai_ddk_lib/" # Download HiAI DDK from https://developer.huawei.com/consumer/cn/hiai/
BUILD_XPU=OFF
BUILD_XTCL=OFF
XPU_SDK_ROOT="$(pwd)/xpu_sdk_lib/"
BUILD_APU=OFF
APU_DDK_ROOT="$(pwd)/apu_sdk_lib/"
BUILD_RKNPU=OFF
RKNPU_DDK_ROOT="$(pwd)/rknpu/"
LITE_WITH_ARM_LANG=OFF
PYTHON_EXECUTABLE_OPTION=""

readonly THIRDPARTY_TAR=https://paddle-inference-dist.bj.bcebos.com/PaddleLite/third-party-05b862.tar.gz

readonly workspace=$PWD

# if operating in mac env, we should expand the maximum file num
os_nmae=`uname -s`
if [ ${os_nmae} == "Darwin" ]; then
   ulimit -n 1024
fi

# for code gen, a source file is generated after a test, but is dependended by some targets in cmake.
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

function build_opt {
    cd $workspace
    prepare_thirdparty
    mkdir -p build.opt
    cd build.opt
    cmake .. -DWITH_LITE=ON \
      -DLITE_ON_MODEL_OPTIMIZE_TOOL=ON \
      -DWITH_TESTING=OFF \
      -DLITE_BUILD_EXTRA=ON \
      -DWITH_MKL=OFF
    make opt -j$NUM_PROC
}

function make_tiny_publish_so {
  local os=$1
  local abi=$2
  local lang=$3
  local android_stl=$4

  cur_dir=$(pwd)
  build_dir=$cur_dir/build.lite.${os}.${abi}.${lang}
  if [ -d $build_dir ]
  then
    rm -rf $build_dir
  fi
  mkdir -p $build_dir
  cd $build_dir

  if [ ${os} == "armlinux" ]; then
    BUILD_JAVA=OFF
  fi
  
  cmake .. \
      ${PYTHON_FLAGS} \
      ${CMAKE_COMMON_OPTIONS} \
      -DWITH_TESTING=OFF \
      -DLITE_WITH_JAVA=$BUILD_JAVA \
      -DLITE_WITH_PYTHON=$BUILD_PYTHON \
      -DLITE_WITH_LOG=$WITH_LOG \
      -DLITE_ON_TINY_PUBLISH=ON \
      -DANDROID_STL_TYPE=$android_stl \
      -DLITE_BUILD_EXTRA=$BUILD_EXTRA \
      -DLITE_WITH_CV=$BUILD_CV \
      -DLITE_WITH_ARM_LANG=$LITE_WITH_ARM_LANG \
      -DLITE_BUILD_TAILOR=$BUILD_TAILOR \
      -DLITE_OPTMODEL_DIR=$OPTMODEL_DIR \
      -DLITE_WITH_NPU=$BUILD_NPU \
      -DNPU_DDK_ROOT=$NPU_DDK_ROOT \
      -DLITE_WITH_XPU=$BUILD_XPU \
      -DLITE_WITH_XTCL=$BUILD_XTCL \
      -DXPU_SDK_ROOT=$XPU_SDK_ROOT \
      -DLITE_WITH_APU=$BUILD_APU \
      -DAPU_DDK_ROOT=$APU_DDK_ROOT \
      -DLITE_WITH_RKNPU=$BUILD_RKNPU \
      -DRKNPU_DDK_ROOT=$RKNPU_DDK_ROOT \
      -DARM_TARGET_OS=${os} -DARM_TARGET_ARCH_ABI=${abi} -DARM_TARGET_LANG=${lang}

  make publish_inference -j$NUM_PROC
  cd - > /dev/null
}

function make_opencl {
  local os=$1
  local abi=$2
  local lang=$3
  #git submodule update --init --recursive
  prepare_thirdparty

  root_dir=$(pwd)
  build_dir=$root_dir/build.lite.${os}.${abi}.${lang}.opencl
  if [ -d $build_directory ]
  then
  rm -rf $build_directory
  fi
  mkdir -p $build_dir
  cd $build_dir
  prepare_workspace $root_dir $build_dir
  prepare_opencl_source_code $root_dir $build_dir
  # $1: ARM_TARGET_OS in "android" , "armlinux"
  # $2: ARM_TARGET_ARCH_ABI in "armv8", "armv7" ,"armv7hf"
  # $3: ARM_TARGET_LANG in "gcc" "clang"
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
      -DLITE_WITH_LOG=$WITH_LOG \
      -DLITE_WITH_CV=$BUILD_CV \
      -DARM_TARGET_OS=$1 -DARM_TARGET_ARCH_ABI=$2 -DARM_TARGET_LANG=$3

    make opencl_clhpp -j$NUM_PROC
    make publish_inference -j$NUM_PROC
}

function make_full_publish_so {
  local os=$1
  local abi=$2
  local lang=$3
  local android_stl=$4

  #git submodule update --init --recursive
  prepare_thirdparty

  root_dir=$(pwd)
  build_directory=$BUILD_DIR/build.lite.${os}.${abi}.${lang}

  if [ -d $build_directory ]
  then
    rm -rf $build_directory
  fi
  mkdir -p $build_directory
  cd $build_directory

  if [ ${os} == "armlinux" ]; then
    BUILD_JAVA=OFF
  fi

  prepare_workspace $root_dir $build_directory
  cmake $root_dir \
      ${PYTHON_FLAGS} \
      ${CMAKE_COMMON_OPTIONS} \
      -DWITH_TESTING=OFF \
      -DLITE_WITH_JAVA=$BUILD_JAVA \
      -DLITE_WITH_PYTHON=$BUILD_PYTHON \
      -DLITE_WITH_LOG=$WITH_LOG \
      -DANDROID_STL_TYPE=$android_stl \
      -DLITE_BUILD_EXTRA=$BUILD_EXTRA \
      -DLITE_WITH_CV=$BUILD_CV \
      -DLITE_WITH_ARM_LANG=$LITE_WITH_ARM_LANG \
      -DLITE_BUILD_TAILOR=$BUILD_TAILOR \
      -DLITE_OPTMODEL_DIR=$OPTMODEL_DIR \
      -DLITE_WITH_NPU=$BUILD_NPU \
      -DNPU_DDK_ROOT=$NPU_DDK_ROOT \
      -DLITE_WITH_XPU=$BUILD_XPU \
      -DLITE_WITH_XTCL=$BUILD_XTCL \
      -DXPU_SDK_ROOT=$XPU_SDK_ROOT \
      -DLITE_WITH_RKNPU=$BUILD_RKNPU \
      -DRKNPU_DDK_ROOT=$RKNPU_DDK_ROOT \
      -DLITE_WITH_TRAIN=$BUILD_TRAIN \
      -DLITE_WITH_APU=$BUILD_APU \
      -DAPU_DDK_ROOT=$APU_DDK_ROOT \
      -DARM_TARGET_OS=${os} -DARM_TARGET_ARCH_ABI=${abi} -DARM_TARGET_LANG=${lang}

  make publish_inference -j$NUM_PROC
  cd - > /dev/null
}

function make_all_tests {
  local os=$1
  local abi=$2
  local lang=$3

  #git submodule update --init --recursive
  prepare_thirdparty
  root_dir=$(pwd)
  build_directory=$BUILD_DIR/build.lite.${os}.${abi}.${lang}
  if [ -d $build_dir ]
  then
    rm -rf $build_dir
  fi
  mkdir -p $build_directory
  cd $build_directory
 
  prepare_workspace $root_dir $build_directory
  cmake $root_dir \
      ${CMAKE_COMMON_OPTIONS} \
      -DWITH_TESTING=ON \
      -DLITE_BUILD_EXTRA=$BUILD_EXTRA \
      -DLITE_WITH_CV=$BUILD_CV \
      -DLITE_WITH_ARM_LANG=$LITE_WITH_ARM_LANG \
      -DLITE_WITH_NPU=$BUILD_NPU \
      -DNPU_DDK_ROOT=$NPU_DDK_ROOT \
      -DLITE_WITH_XPU=$BUILD_XPU \
      -DLITE_WITH_XTCL=$BUILD_XTCL \
      -DXPU_SDK_ROOT=$XPU_SDK_ROOT \
      -DLITE_WITH_APU=$BUILD_APU \
      -DAPU_DDK_ROOT=$APU_DDK_ROOT \
      -DLITE_WITH_RKNPU=$BUILD_RKNPU \
      -DRKNPU_DDK_ROOT=$RKNPU_DDK_ROOT \
      -DARM_TARGET_OS=${os} -DARM_TARGET_ARCH_ABI=${abi} -DARM_TARGET_LANG=${lang}

  make lite_compile_deps -j$NUM_PROC
  cd - > /dev/null
}

function make_ios {
    local os=$1
    local abi=$2
    build_dir=build.ios.${os}.${abi}
    echo "building ios target into $build_dir"
    echo "target os: $os"
    echo "target abi: $abi"
    mkdir -p ${build_dir}
    cd ${build_dir}
    GEN_CODE_PATH_PREFIX=lite/gen_code
    mkdir -p ./${GEN_CODE_PATH_PREFIX}
    touch ./${GEN_CODE_PATH_PREFIX}/__generated_code__.cc

    cmake .. \
            -DWITH_GPU=OFF \
            -DWITH_MKL=OFF \
            -DWITH_LITE=ON \
            -DLITE_WITH_CUDA=OFF \
            -DLITE_WITH_X86=OFF \
            -DLITE_WITH_ARM=ON \
            -DWITH_TESTING=OFF \
            -DLITE_WITH_JAVA=OFF \
            -DLITE_WITH_LOG=ON \
            -DLITE_ON_TINY_PUBLISH=ON \
            -DLITE_WITH_OPENMP=OFF \
            -DWITH_ARM_DOTPROD=OFF \
            -DLITE_BUILD_TAILOR=$BUILD_TAILOR \
            -DLITE_OPTMODEL_DIR=$OPTMODEL_DIR \
            -DLITE_WITH_LIGHT_WEIGHT_FRAMEWORK=ON \
            -DARM_TARGET_ARCH_ABI=$abi \
            -DLITE_BUILD_EXTRA=$BUILD_EXTRA \
            -DLITE_WITH_CV=$BUILD_CV \
            -DARM_TARGET_OS=$os

    make publish_inference -j$NUM_PROC
    cd -
}

function make_cuda {
  prepare_thirdparty

  root_dir=$(pwd)
  build_directory=$BUILD_DIR/build_cuda

  if [ -d $build_directory ]
  then
    rm -rf $build_directory
  fi
  mkdir -p $build_directory
  cd $build_directory

  prepare_workspace $root_dir $build_directory

  cmake ..  -DWITH_MKL=OFF       \
            -DLITE_WITH_CUDA=ON  \
            -DWITH_MKLDNN=OFF    \
            -DLITE_WITH_X86=OFF  \
            -DLITE_WITH_PROFILE=OFF \
            -DWITH_LITE=ON \
            -DLITE_WITH_LIGHT_WEIGHT_FRAMEWORK=OFF \
            -DWITH_TESTING=OFF \
            -DLITE_WITH_ARM=OFF \
            -DLITE_WITH_STATIC_CUDA=OFF \
            -DLITE_WITH_PYTHON=${BUILD_PYTHON} \
            -DLITE_BUILD_EXTRA=ON \
            -DLITE_WITH_XPU=$BUILD_XPU \
            -DLITE_WITH_XTCL=$BUILD_XTCL \
            -DXPU_SDK_ROOT=$XPU_SDK_ROOT
 
  make publish_inference -j$NUM_PROC
  cd -
}

function make_x86 {
  prepare_thirdparty

  root_dir=$(pwd)
  build_directory=$BUILD_DIR/build.lite.x86

  if [ -d $build_directory ]
  then
    rm -rf $build_directory
  fi
  mkdir -p $build_directory
  cd $build_directory

  prepare_workspace $root_dir $build_directory

  cmake ..  -DWITH_MKL=ON       \
            -DWITH_MKLDNN=OFF    \
            -DLITE_WITH_X86=ON  \
            -DLITE_WITH_PROFILE=OFF \
            -DWITH_LITE=ON \
            -DLITE_WITH_LIGHT_WEIGHT_FRAMEWORK=OFF \
            -DLITE_WITH_ARM=OFF \
            -DWITH_GPU=OFF \
            -DLITE_WITH_PYTHON=${BUILD_PYTHON} \
            -DLITE_BUILD_EXTRA=ON \
            -DLITE_WITH_XPU=$BUILD_XPU \
            -DLITE_WITH_XTCL=$BUILD_XTCL \
            -DXPU_SDK_ROOT=$XPU_SDK_ROOT \
            -DCMAKE_BUILD_TYPE=Release \
            $PYTHON_EXECUTABLE_OPTION

  make publish_inference -j$NUM_PROC
  cd -
}

function print_usage {
    set +x
    echo -e "\nUSAGE:"
    echo
    echo "----------------------------------------"
    echo -e "compile tiny publish so lib:"
    echo -e "for android:"
    echo -e "   ./build.sh --arm_os=<os> --arm_abi=<abi> --arm_lang=<lang> --android_stl=<stl> tiny_publish"
    echo -e "for ios:"
    echo -e "   ./build.sh --arm_os=<os> --arm_abi=<abi> ios"
    echo
    echo -e "compile full publish so lib (ios not support):"
    echo -e "   ./build.sh --arm_os=<os> --arm_abi=<abi> --arm_lang=<lang> --android_stl=<stl> full_publish"
    echo
    echo -e "compile all arm tests (ios not support):"
    echo -e "   ./build.sh --arm_os=<os> --arm_abi=<abi> --arm_lang=<lang> test"
    echo
    echo -e "optional argument:"
    echo -e "--with_log: (OFF|ON); controls whether to print log information, default is ON"
    echo -e "--build_extra: (OFF|ON); controls whether to publish extra operators and kernels for (sequence-related model such as OCR or NLP)"
    echo -e "--build_train: (OFF|ON); controls whether to publish training operators and kernels, build_train is only for full_publish library now"
    echo -e "--build_python: (OFF|ON); controls whether to publish python api lib (ANDROID and IOS is not supported)"
    echo -e "--build_java: (OFF|ON); controls whether to publish java api lib (Only ANDROID is supported)"
    echo -e "--build_dir: directory for building"
    echo
    echo -e "argument choices:"
    echo -e "--arm_os:\t android|ios|ios64"
    echo -e "--arm_abi:\t armv8|armv7"
    echo -e "--arm_lang:\t only support gcc now, clang will be supported in future.(for android)"
    echo -e "--android_stl:\t c++_static|c++_shared (for android)"
    echo
    echo -e "tasks:"
    echo
    echo -e "tiny_publish: a small library for deployment."
    echo -e "full_publish: a full library for debug and test."
    echo -e "test: produce all the unittests."
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
            --arm_os=*)
                ARM_OS="${i#*=}"
                shift
                ;;
            --arm_abi=*)
                ARM_ABI="${i#*=}"
                shift
                ;;
            --arm_lang=*)
                ARM_LANG="${i#*=}"
                shift
                ;;
            --android_stl=*)
                ANDROID_STL="${i#*=}"
                shift
                ;;
            --build_extra=*)
                BUILD_EXTRA="${i#*=}"
                shift
                ;;
            --build_train=*)
                BUILD_TRAIN="${i#*=}"
                shift
                ;;
            --build_cv=*)
                BUILD_CV="${i#*=}"
                shift
                ;;
            --build_python=*)
                BUILD_PYTHON="${i#*=}"
                shift
                ;;
            --build_java=*)
                BUILD_JAVA="${i#*=}"
                shift
                ;;
            --build_dir=*)
                BUILD_DIR="${i#*=}"
                shift
		;;
            --opt_model_dir=*)
                OPTMODEL_DIR="${i#*=}"
                shift
                ;;
            --build_tailor=*)
                BUILD_TAILOR="${i#*=}"
                shift
                ;;
            --with_log=*)
                WITH_LOG="${i#*=}"
                shift
                ;;
            --build_npu=*)
                BUILD_NPU="${i#*=}"
                shift
                ;;
           --npu_ddk_root=*)
                NPU_DDK_ROOT="${i#*=}"
                shift
                ;;
            --build_xpu=*)
                BUILD_XPU="${i#*=}"
                shift
                ;;
            --build_xtcl=*)
                BUILD_XTCL="${i#*=}"
                shift
                ;;
            --xpu_sdk_root=*)
                XPU_SDK_ROOT="${i#*=}"
                shift
                ;;
            --python_executable=*)
                PYTHON_EXECUTABLE_OPTION="-DPYTHON_EXECUTABLE=${i#*=}"
                shift
                ;;
            --build_apu=*)
                BUILD_APU="${i#*=}"
                shift
                ;;
           --apu_ddk_root=*)
                APU_DDK_ROOT="${i#*=}"
                shift
                ;;
            --build_rknpu=*)
                BUILD_RKNPU="${i#*=}"
                shift
                ;;
            --rknpu_ddk_root=*)
                RKNPU_DDK_ROOT="${i#*=}"
                shift
                ;;
            tiny_publish)
                make_tiny_publish_so $ARM_OS $ARM_ABI $ARM_LANG $ANDROID_STL 
                shift
                ;;
            full_publish)
                make_full_publish_so $ARM_OS $ARM_ABI $ARM_LANG $ANDROID_STL 
                shift
                ;;
            test)
                make_all_tests $ARM_OS $ARM_ABI $ARM_LANG
                shift
                ;;
            ios)
                make_ios $ARM_OS $ARM_ABI
                shift
                ;;
            build_optimize_tool)
                build_opt
                shift
                ;;
            opencl)
                make_opencl $ARM_OS $ARM_ABI $ARM_LANG
                shift
                ;;
            cuda)
                make_cuda
                shift
                ;;
            x86)
               make_x86
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
