#!/bin/bash
set -e

readonly CMAKE_COMMON_OPTIONS="-DWITH_MKL=OFF \
                               -DLITE_WITH_X86=OFF \
                               -DLITE_WITH_ARM=ON"

readonly NUM_PROC=${LITE_BUILD_THREADS:-8}

# global variables
CMAKE_EXTRA_OPTIONS=""
BUILD_EXTRA=ON
BUILD_TRAIN=OFF
BUILD_JAVA=ON
BUILD_PYTHON=OFF
BUILD_DIR=$(pwd)
OPTMODEL_DIR=""
BUILD_TAILOR=OFF
BUILD_THREAD_POOL=OFF
BUILD_CV=OFF
WITH_LOG=ON
WITH_MKL=ON
WITH_OPENCL=OFF
WITH_STATIC_MKL=OFF
WITH_AVX=ON
WITH_EXCEPTION=OFF
WITH_PROFILE=OFF
WITH_PRECISION_PROFILE=OFF
WITH_LTO=OFF
BUILD_ARM82_FP16=OFF
BUILD_ARM82_INT8_SDOT=OFF
# controls whether to support SVE2 instructions, default is OFF
WITH_ARM8_SVE2=OFF
BUILD_XPU=OFF
BUILD_XTCL=OFF
XPU_SDK_ROOT=""
XPU_SDK_URL=""
XPU_SDK_ENV=""
BUILD_APU=OFF
APU_DDK_ROOT="$(pwd)/apu_sdk_lib/"
BUILD_RKNPU=OFF
RKNPU_DDK_ROOT="$(pwd)/rknpu/"
# default installation path, ensure acllib/atc/opp directories are all in this root dir
PYTHON_EXECUTABLE_OPTION=""
IOS_DEPLOYMENT_TARGET=9.0
WITH_NODE_RAW_FS=OFF
# min android api level
MIN_ANDROID_API_LEVEL_ARMV7=16
MIN_ANDROID_API_LEVEL_ARMV8=21
# android api level, which can also be set to a specific number
ANDROID_API_LEVEL="Default"
CMAKE_API_LEVEL_OPTIONS=""

# url that stores third-party tar.gz file to accelerate third-party lib installation
readonly THIRDPARTY_URL=https://paddlelite-data.bj.bcebos.com/third_party_libs/
readonly THIRDPARTY_TAR=third-party-651c7c4.tar.gz
readonly workspace=$PWD

function readlinkf() {
    perl -MCwd -e 'print Cwd::abs_path shift' "$1";
}

# if operating in mac env, we should expand the maximum file num
os_name=`uname -s`
if [ ${os_name} == "Darwin" ]; then
   ulimit -n 1024
fi

# for code gen, a source file is generated after a test, but is dependended by some targets in cmake.
# here we fake an empty file to make cmake works.
function prepare_workspace {
    local root_dir=$1
    local build_dir=$2
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
    if [ ! -d $workspace/third-party -o -f $workspace/$THIRDPARTY_TAR ]; then
        rm -rf $workspace/third-party

        if [ ! -f $workspace/$THIRDPARTY_TAR ]; then
            wget $THIRDPARTY_URL/$THIRDPARTY_TAR
        fi
        tar xzf $THIRDPARTY_TAR
    else
        git submodule update --init --recursive
    fi
}

function set_android_api_level {
  # android api level for android version
  if [ "${ARM_ABI}" == "armv7" ]; then
      MIN_ANDROID_API_LEVEL=${MIN_ANDROID_API_LEVEL_ARMV7}
  else
      MIN_ANDROID_API_LEVEL=${MIN_ANDROID_API_LEVEL_ARMV8}
  fi
  if [ "${ANDROID_API_LEVEL}" == "Default" ]; then
      CMAKE_API_LEVEL_OPTIONS=""
  elif [ ${ANDROID_API_LEVEL} -ge ${MIN_ANDROID_API_LEVEL} ]; then
      CMAKE_API_LEVEL_OPTIONS="-DANDROID_NATIVE_API_LEVEL=${ANDROID_API_LEVEL}"
  else
      echo "Error: ANDROID_API_LEVEL should be no less than ${MIN_ANDROID_API_LEVEL} on ${ARM_ABI}."
      exit 1
  fi
}

function build_opt {
    cd $workspace
    prepare_thirdparty
    rm -rf build.opt
    mkdir -p build.opt
    cd build.opt
    opt_arch=$(echo `uname -p`)
    with_x86=OFF
    if [ $opt_arch == "aarch64" ]; then
        with_x86=OFF
    else
       with_x86=ON
    fi
    cmake .. \
      -DLITE_ON_MODEL_OPTIMIZE_TOOL=ON \
      -DWITH_TESTING=OFF \
      -DLITE_BUILD_EXTRA=ON \
      -DLITE_WITH_X86=${with_x86} \
      -DWITH_MKL=OFF
    make opt -j$NUM_PROC
}

function build_opt_wasm {
    cd $workspace
    prepare_thirdparty
    cd third-party/protobuf-host
    git apply $workspace/cmake/protobuf-host-patch || true
    cd $workspace
    mkdir -p build-protoc
    cd build-protoc
    cmake -Dprotobuf_BUILD_TESTS=OFF ../third-party/protobuf-host/cmake
    make protoc -j$NUM_PROC
    cd ..
    mkdir -p build.opt.wasm
    cd build.opt.wasm
    emcmake cmake .. \
      -DLITE_ON_MODEL_OPTIMIZE_TOOL=ON \
      -DWITH_TESTING=OFF \
      -DLITE_BUILD_EXTRA=ON \
      -DWITH_MKL=OFF \
      -DLITE_WITH_X86=OFF \
      -DLITE_WITH_OPENMP=OFF \
      -DPROTOBUF_PROTOC_EXECUTABLE=`pwd`/../build-protoc/protoc \
      -DWITH_NODE_RAW_FS=$1
    emmake make opt -j$NUM_PROC
    cd ../third-party/protobuf-host
    git reset --hard HEAD
}

function make_tiny_publish_so {
  local os=$1
  local abi=$2
  local lang=$3
  local android_stl=$4

  cur_dir=$(pwd)
  build_dir=$cur_dir/build.lite.${os}.${abi}.${lang}
  if [ ! -d third-party ]; then
    git checkout third-party
  fi
  if [ "${BUILD_ARM82_FP16}" == "ON" ]; then
      TOOLCHAIN=clang
      build_dir=$build_dir".armv82_fp16"
  fi
  if [ "${WITH_ARM8_SVE2}" == "ON" ]; then
      TOOLCHAIN=clang
      build_dir=$build_dir".armv8_sve2"
  fi

  if [ -d $build_dir ]
  then
    rm -rf $build_dir
  fi
  mkdir -p $build_dir
  cd $build_dir

  if [ ${os} == "armlinux" ]; then
    BUILD_JAVA=OFF
  fi

  if [ ${os} == "android" ]; then
    set_android_api_level
    CMAKE_EXTRA_OPTIONS=${CMAKE_EXTRA_OPTIONS}" "${CMAKE_API_LEVEL_OPTIONS}
  fi

  cmake .. \
      ${PYTHON_FLAGS} \
      ${CMAKE_COMMON_OPTIONS} \
      ${CMAKE_EXTRA_OPTIONS} \
      -DWITH_TESTING=OFF \
      -DLITE_WITH_JAVA=$BUILD_JAVA \
      -DLITE_WITH_PYTHON=$BUILD_PYTHON \
      -DLITE_WITH_LOG=$WITH_LOG \
      -DLITE_WITH_EXCEPTION=$WITH_EXCEPTION \
      -DLITE_ON_TINY_PUBLISH=ON \
      -DANDROID_STL_TYPE=$android_stl \
      -DLITE_BUILD_EXTRA=$BUILD_EXTRA \
      -DLITE_WITH_CV=$BUILD_CV \
      -DLITE_BUILD_TAILOR=$BUILD_TAILOR \
      -DLITE_OPTMODEL_DIR=$OPTMODEL_DIR \
      -DLITE_WITH_XPU=$BUILD_XPU \
      -DLITE_WITH_XTCL=$BUILD_XTCL \
      -DXPU_SDK_ROOT=$XPU_SDK_ROOT \
      -DXPU_SDK_URL=$XPU_SDK_URL \
      -DXPU_SDK_ENV=$XPU_SDK_ENV \
      -DLITE_WITH_APU=$BUILD_APU \
      -DAPU_DDK_ROOT=$APU_DDK_ROOT \
      -DLITE_WITH_RKNPU=$BUILD_RKNPU \
      -DRKNPU_DDK_ROOT=$RKNPU_DDK_ROOT \
      -DLITE_WITH_ARM82_FP16=$BUILD_ARM82_FP16 \
      -DLITE_WITH_ARM8_SVE2=$WITH_ARM8_SVE2 \
      -DLITE_WITH_ARM82_INT8_SDOT=$BUILD_ARM82_INT8_SDOT \
      -DLITE_THREAD_POOL=$BUILD_THREAD_POOL \
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

  if [ ${os} == "android" ]; then
    set_android_api_level
  fi

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
      ${CMAKE_API_LEVEL_OPTIONS} \
      -DLITE_WITH_OPENCL=ON \
      -DWITH_MKL=OFF \
      -DLITE_WITH_X86=OFF \
      -DLITE_WITH_ARM=ON \
      -DWITH_ARM_DOTPROD=ON   \
      -DLITE_ON_TINY_PUBLISH=ON \
      -DWITH_TESTING=OFF \
      -DLITE_BUILD_EXTRA=$BUILD_EXTRA \
      -DLITE_WITH_LOG=$WITH_LOG \
      -DLITE_WITH_EXCEPTION=$WITH_EXCEPTION \
      -DLITE_WITH_CV=$BUILD_CV \
      -DARM_TARGET_OS=$1 -DARM_TARGET_ARCH_ABI=$2 -DARM_TARGET_LANG=$3

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
  if [ "${BUILD_ARM82_FP16}" == "ON" ]; then
      TOOLCHAIN=clang
      build_directory=$build_directory".armv82_fp16"
  fi
  if [ "${WITH_ARM8_SVE2}" == "ON" ]; then
      TOOLCHAIN=clang
      build_directory=$build_directory".armv8_sve2"
  fi

  if [ -d $build_directory ]
  then
    rm -rf $build_directory
  fi
  mkdir -p $build_directory
  cd $build_directory

  if [ ${os} == "armlinux" ]; then
    BUILD_JAVA=OFF
  fi

  if [ ${os} == "android" ]; then
    set_android_api_level
    CMAKE_EXTRA_OPTIONS=${CMAKE_EXTRA_OPTIONS}" "${CMAKE_API_LEVEL_OPTIONS}
  fi

  prepare_workspace $root_dir $build_directory
  cmake $root_dir \
      ${PYTHON_FLAGS} \
      ${CMAKE_COMMON_OPTIONS} \
      ${CMAKE_EXTRA_OPTIONS} \
      -DWITH_TESTING=OFF \
      -DLITE_WITH_JAVA=$BUILD_JAVA \
      -DLITE_WITH_PYTHON=$BUILD_PYTHON \
      -DLITE_WITH_LOG=$WITH_LOG \
      -DLITE_WITH_EXCEPTION=$WITH_EXCEPTION \
      -DLITE_WITH_PROFILE=${WITH_PROFILE} \
      -DLITE_WITH_PRECISION_PROFILE=${WITH_PRECISION_PROFILE} \
      -DLITE_WITH_LTO=${WITH_LTO} \
      -DANDROID_STL_TYPE=$android_stl \
      -DLITE_BUILD_EXTRA=$BUILD_EXTRA \
      -DLITE_WITH_CV=$BUILD_CV \
      -DLITE_BUILD_TAILOR=$BUILD_TAILOR \
      -DLITE_OPTMODEL_DIR=$OPTMODEL_DIR \
      -DLITE_WITH_XPU=$BUILD_XPU \
      -DLITE_WITH_XTCL=$BUILD_XTCL \
      -DXPU_SDK_ROOT=$XPU_SDK_ROOT \
      -DXPU_SDK_URL=$XPU_SDK_URL \
      -DXPU_SDK_ENV=$XPU_SDK_ENV \
      -DLITE_WITH_RKNPU=$BUILD_RKNPU \
      -DRKNPU_DDK_ROOT=$RKNPU_DDK_ROOT \
      -DLITE_WITH_TRAIN=$BUILD_TRAIN \
      -DLITE_WITH_APU=$BUILD_APU \
      -DAPU_DDK_ROOT=$APU_DDK_ROOT \
      -DLITE_WITH_ARM82_FP16=$BUILD_ARM82_FP16 \
      -DLITE_WITH_ARM8_SVE2=$WITH_ARM8_SVE2 \
      -DLITE_WITH_ARM82_INT8_SDOT=$BUILD_ARM82_INT8_SDOT \
      -DARM_TARGET_OS=${os} -DARM_TARGET_ARCH_ABI=${abi} -DARM_TARGET_LANG=${lang}

  make publish_inference -j$NUM_PROC
  cd - > /dev/null
}

function set_benchmark_options {
  BUILD_EXTRA=ON
  WITH_EXCEPTION=ON
  BUILD_JAVA=OFF
  WITH_OPENCL=ON
  if [ ${WITH_PROFILE} == "ON" ] || [ ${WITH_PRECISION_PROFILE} == "ON" ]; then
    WITH_LOG=ON
  else
    WITH_LOG=OFF
  fi
}

function make_all_tests {
  local os=$1
  local abi=$2
  local lang=$3

  #git submodule update --init --recursive
  prepare_thirdparty
  root_dir=$(pwd)
  build_directory=$BUILD_DIR/build.lite.${os}.${abi}.${lang}
  if [ $4 == "benchmark" ]; then
    set_benchmark_options
    build_directory=$build_directory".benchmark"
  fi
  if [ "${BUILD_ARM82_FP16}" == "ON" ]; then
      TOOLCHAIN=clang
      build_directory=$build_directory".armv82_fp16"
  fi
  if [ "${WITH_ARM8_SVE2}" == "ON" ]; then
      TOOLCHAIN=clang
      build_directory=$build_directory".armv8_sve2"
  fi

  if [ -d $build_directory ]
  then
    rm -rf $build_directory
  fi
  mkdir -p $build_directory

  cd $build_directory
  if [ ${os} == "android" ]; then
    set_android_api_level
    CMAKE_EXTRA_OPTIONS=${CMAKE_EXTRA_OPTIONS}" "${CMAKE_API_LEVEL_OPTIONS}
  fi

  prepare_workspace $root_dir $build_directory
  cmake $root_dir \
      ${CMAKE_COMMON_OPTIONS} \
      ${CMAKE_EXTRA_OPTIONS} \
      -DWITH_TESTING=ON \
      -DLITE_WITH_LOG=${WITH_LOG} \
      -DLITE_WITH_PROFILE=${WITH_PROFILE} \
      -DLITE_WITH_LTO=${WITH_LTO} \
      -DLITE_WITH_PRECISION_PROFILE=${WITH_PRECISION_PROFILE} \
      -DLITE_BUILD_EXTRA=$BUILD_EXTRA \
      -DLITE_WITH_CV=$BUILD_CV \
      -DLITE_WITH_OPENCL=$WITH_OPENCL \
      -DLITE_WITH_XPU=$BUILD_XPU \
      -DLITE_WITH_XTCL=$BUILD_XTCL \
      -DXPU_SDK_ROOT=$XPU_SDK_ROOT \
      -DXPU_SDK_URL=$XPU_SDK_URL \
      -DXPU_SDK_ENV=$XPU_SDK_ENV \
      -DLITE_WITH_APU=$BUILD_APU \
      -DAPU_DDK_ROOT=$APU_DDK_ROOT \
      -DLITE_WITH_RKNPU=$BUILD_RKNPU \
      -DRKNPU_DDK_ROOT=$RKNPU_DDK_ROOT \
      -DLITE_WITH_ARM82_FP16=$BUILD_ARM82_FP16 \
      -DLITE_WITH_ARM8_SVE2=$WITH_ARM8_SVE2 \
      -DLITE_WITH_ARM82_INT8_SDOT=$BUILD_ARM82_INT8_SDOT \
      -DARM_TARGET_OS=${os} -DARM_TARGET_ARCH_ABI=${abi} -DARM_TARGET_LANG=${lang}

  if [ $4 == "benchmark" ]; then
    make benchmark_bin -j$NUM_PROC
  else
    make lite_compile_deps -j$NUM_PROC
  fi
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
            -DWITH_MKL=OFF \
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
            -DARM_TARGET_ARCH_ABI=$abi \
            -DLITE_BUILD_EXTRA=$BUILD_EXTRA \
            -DLITE_WITH_CV=$BUILD_CV \
            -DDEPLOYMENT_TARGET=${IOS_DEPLOYMENT_TARGET} \
            -DARM_TARGET_OS=$os

    make publish_inference -j$NUM_PROC
    cd -
}

function make_x86 {
  prepare_thirdparty

  root_dir=$(pwd)
  build_directory=$BUILD_DIR/build.lite.x86

  if [ ${WITH_OPENCL} == "ON" ]; then
    BUILD_EXTRA=ON
    build_directory=$BUILD_DIR/build.lite.x86.opencl
    prepare_opencl_source_code $root_dir $build_directory
  fi

  if [ ${BUILD_PYTHON} == "ON" ]; then
    BUILD_EXTRA=ON
  fi

  if [ -d $build_directory ]
  then
    rm -rf $build_directory
  fi
  mkdir -p $build_directory
  cd $build_directory

  prepare_workspace $root_dir $build_directory

  cmake $root_dir  -DWITH_MKL=${WITH_MKL}  \
            -DWITH_STATIC_MKL=${WITH_STATIC_MKL}  \
            -DWITH_TESTING=OFF \
            -DWITH_AVX=${WITH_AVX} \
            -DWITH_MKLDNN=OFF    \
            -DLITE_WITH_X86=ON  \
            -DLITE_WITH_ARM=OFF \
            -DLITE_WITH_OPENCL=${WITH_OPENCL} \
            -DLITE_WITH_PYTHON=${BUILD_PYTHON} \
            -DLITE_BUILD_EXTRA=${BUILD_EXTRA} \
            -DLITE_BUILD_TAILOR=${BUILD_TAILOR} \
            -DLITE_OPTMODEL_DIR=${OPTMODEL_DIR} \
            -DLITE_WITH_LOG=${WITH_LOG} \
            -DLITE_WITH_EXCEPTION=$WITH_EXCEPTION \
            -DLITE_WITH_PROFILE=${WITH_PROFILE} \
            -DLITE_WITH_PRECISION_PROFILE=${WITH_PRECISION_PROFILE} \
            -DLITE_WITH_LTO=${WITH_LTO} \
            -DLITE_WITH_XPU=$BUILD_XPU \
            -DLITE_WITH_XTCL=$BUILD_XTCL \
            -DXPU_SDK_ROOT=$XPU_SDK_ROOT \
            -DXPU_SDK_URL=$XPU_SDK_URL \
            -DXPU_SDK_ENV=$XPU_SDK_ENV \
            -DCMAKE_BUILD_TYPE=Release \
            -DPY_VERSION=$PY_VERSION \
            $PYTHON_EXECUTABLE_OPTION

  make publish_inference -j$NUM_PROC
  cd -
}

function make_x86_tests {
  prepare_thirdparty

  root_dir=$(pwd)
  build_directory=$BUILD_DIR/build.lite.x86_tests

  if [ $1 == "benchmark" ]; then
    set_benchmark_options
    if [ ${os_name} == "Linux" ]; then
      # Turn off opencl. Additional third party library need to be installed on
      # Linux. Otherwise opencl is not supported on Linux.
      WITH_OPENCL=OFF
    fi
  fi

  if [ ${WITH_OPENCL} == "ON" ]; then
    BUILD_EXTRA=ON
    build_directory=$BUILD_DIR/build.lite.x86.opencl
    prepare_opencl_source_code $root_dir $build_directory
  fi

  if [ ${BUILD_PYTHON} == "ON" ]; then
    BUILD_EXTRA=ON
  fi

  if [ -d $build_directory ]
  then
    rm -rf $build_directory
  fi
  mkdir -p $build_directory
  cd $build_directory

  prepare_workspace $root_dir $build_directory

  cmake $root_dir  -DWITH_MKL=${WITH_MKL}  \
            -DWITH_STATIC_MKL=${WITH_STATIC_MKL}  \
            -DWITH_TESTING=ON \
            -DLITE_WITH_PROFILE=${WITH_PROFILE} \
            -DLITE_WITH_PRECISION_PROFILE=${WITH_PRECISION_PROFILE} \
            -DWITH_AVX=${WITH_AVX} \
            -DWITH_MKLDNN=OFF   \
            -DLITE_WITH_X86=ON  \
            -DLITE_WITH_ARM=OFF \
            -DLITE_WITH_OPENCL=${WITH_OPENCL} \
            -DLITE_WITH_PYTHON=${BUILD_PYTHON} \
            -DLITE_BUILD_EXTRA=${BUILD_EXTRA} \
            -DLITE_BUILD_TAILOR=${BUILD_TAILOR} \
            -DLITE_OPTMODEL_DIR=${OPTMODEL_DIR} \
            -DLITE_WITH_LOG=${WITH_LOG} \
            -DLITE_WITH_EXCEPTION=$WITH_EXCEPTION \
            -DLITE_WITH_LTO=${WITH_LTO} \
            -DLITE_WITH_XPU=$BUILD_XPU \
            -DLITE_WITH_XTCL=$BUILD_XTCL \
            -DXPU_SDK_ROOT=$XPU_SDK_ROOT \
            -DXPU_SDK_URL=$XPU_SDK_URL \
            -DXPU_SDK_ENV=$XPU_SDK_ENV \
            -DCMAKE_BUILD_TYPE=Debug \
            -DPY_VERSION=$PY_VERSION \
            $PYTHON_EXECUTABLE_OPTION

  if [ $1 == "benchmark" ]; then
    make benchmark_bin -j$NUM_PROC
  else
    make lite_compile_deps -j$NUM_PROC
  fi
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
    echo -e "compile benchmark_bin for android:"
    echo -e "   ./build.sh --arm_os=<os> --arm_abi=<abi> --arm_lang=<lang> --with_profile=<ON|OFF> benchmark"
    echo
    echo -e "compile benchmark_bin for x86 linux:"
    echo -e "   ./build.sh --with_avx=ON --with_profile=<ON|OFF> x86_benchmark"
    echo
    echo -e "optional argument:"
    echo -e "--with_log: (OFF|ON); controls whether to print log information, default is ON"
    echo -e "--with_exception: (OFF|ON); controls whether to throw the exception when error occurs, default is OFF"
    echo -e "--build_extra: (OFF|ON); controls whether to publish extra operators and kernels for (sequence-related model such as OCR or NLP)"
    echo -e "--with_profile: (OFF|ON); controls whether to support time profile, default is OFF"
    echo -e "--with_precision_profile: (OFF|ON); controls whether to support precision profile, default is OFF"
    echo -e "--build_train: (OFF|ON); controls whether to publish training operators and kernels, build_train is only for full_publish library now"
    echo -e "--build_python: (OFF|ON); controls whether to publish python api lib (ANDROID and IOS is not supported)"
    echo -e "--build_java: (OFF|ON); controls whether to publish java api lib (Only ANDROID is supported)"
    echo -e "--build_dir: directory for building"
    echo -e "--ios_deployment_target: (default: 9.0); Set the minimum compatible system version for ios deployment."
    echo -e "|     --with_arm8_sve2: (OFF|ON); controls whether to include SVE2 kernels, default is OFF                                             |"
    echo -e "|                                  warning: when --with_arm8_sve2=ON, NDK version need >= r23, arch will be set as armv8.              |"
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
            --android_api_level=*)
                ANDROID_API_LEVEL="${i#*=}"
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
                OPTMODEL_DIR=$(readlinkf $OPTMODEL_DIR)
                shift
                ;;
            --build_tailor=*)
                BUILD_TAILOR="${i#*=}"
                shift
                ;;
            --build_thread_pool=*)
                BUILD_THREAD_POOL="${i#*=}"
                shift
                ;;
            --with_log=*)
                WITH_LOG="${i#*=}"
                shift
                ;;
            --with_mkl=*)
                WITH_MKL="${i#*=}"
                shift
                ;;
            --with_static_mkl=*)
                WITH_STATIC_MKL="${i#*=}"
                shift
                ;;
            --with_avx=*)
                WITH_AVX="${i#*=}"
                shift
                ;;
            --with_exception=*)
                WITH_EXCEPTION="${i#*=}"
                if [[ $WITH_EXCEPTION == "ON" && $ARM_OS=="android" && $ARM_ABI == "armv7" && $ARM_LANG != "clang" ]]; then
                     set +x
                     echo
                     echo -e "error: only clang provide C++ exception handling support for 32-bit ARM."
                     echo
                     exit 1
                fi
                shift
                ;;
            --with_profile=*)
                WITH_PROFILE="${i#*=}"
                shift
                ;;
            --with_precision_profile=*)
                WITH_PRECISION_PROFILE="${i#*=}"
                shift
                ;;
            --with_lto=*)
                WITH_LTO="${i#*=}"
                shift
                ;;
            --build_arm82_fp16=*)
                BUILD_ARM82_FP16="${i#*=}"
                shift
                ;;
            --build_arm8_sve2=*)
                 WITH_ARM8_SVE2="${i#*=}"
                 shift
                 ;;
            --build_arm82_int8_sdot=*)
                BUILD_ARM82_INT8_SDOT="${i#*=}"
                shift
                ;;
            --build_opencl=*)
                WITH_OPENCL="${i#*=}"
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
                XPU_SDK_ROOT=${i#*=}
                if [ -n "${XPU_SDK_ROOT}" ]; then
                    XPU_SDK_ROOT=$(readlink -f ${XPU_SDK_ROOT})
                fi
                shift
                ;;
            --xpu_sdk_url=*)
                XPU_SDK_URL="${i#*=}"
                shift
                ;;
            --xpu_sdk_env=*)
                XPU_SDK_ENV="${i#*=}"
                shift
                ;;
            --python_executable=*)
                PYTHON_EXECUTABLE_OPTION="-DPYTHON_EXECUTABLE=${i#*=}"
                shift
                ;;
            --python_version=*)
                PY_VERSION="${i#*=}"
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
            --ios_deployment_target=*)
                IOS_DEPLOYMENT_TARGET="${i#*=}"
                shift
                ;;
            --with_node_raw_fs=*)
                WITH_NODE_RAW_FS="${i#*=}"
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
            benchmark)
                make_all_tests $ARM_OS $ARM_ABI $ARM_LANG benchmark
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
            build_optimize_tool_wasm)
                build_opt_wasm $WITH_NODE_RAW_FS
                shift
                ;;
            opencl)
                make_opencl $ARM_OS $ARM_ABI $ARM_LANG
                shift
                ;;
            x86)
               make_x86
               shift
               ;;
            test_x86)
               make_x86_tests
               shift
               ;;
            x86_benchmark)
               make_x86_tests benchmark
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
