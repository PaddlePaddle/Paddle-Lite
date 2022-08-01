#!/bin/bash
set -e
set +x
#####################################################################################################
# 1. global variables, you can change them according to your requirements
#####################################################################################################
# armv7 or armv8, default armv8.
ARCH=armv8
# c++_static or c++_shared, default c++_static.
ANDROID_STL=c++_static
# min android api level
MIN_ANDROID_API_LEVEL_ARMV7=16
MIN_ANDROID_API_LEVEL_ARMV8=21
# android api level, which can also be set to a specific number
ANDROID_API_LEVEL="Default"
# gcc or clang, default gcc.
TOOLCHAIN=gcc
# ON or OFF, default OFF.
WITH_EXTRA=OFF
# ON or OFF, default ON.
WITH_JAVA=ON
# ON or OFF, default is OFF
WITH_STATIC_LIB=OFF
# controls whether to compile cv functions into lib, default is OFF.
WITH_CV=OFF
# controls whether to hide log information, default is ON.
WITH_LOG=ON
# controls whether to throw the exception when error occurs, default is OFF
WITH_EXCEPTION=OFF
# controls whether to include FP16 kernels, default is OFF
BUILD_ARM82_FP16=OFF
# controls whether to support SVE2 instructions, default is OFF
WITH_ARM8_SVE2=OFF
WITH_ARM_DOTPROD=ON
# options of striping lib according to input model.
OPTMODEL_DIR=""
WITH_STRIP=OFF
WITH_THREAD_POOL=OFF
# options of compiling NPU lib.
WITH_HUAWEI_KIRIN_NPU=OFF
HUAWEI_KIRIN_NPU_SDK_ROOT="$(pwd)/ai_ddk_lib/" # Download HiAI DDK from https://developer.huawei.com/consumer/cn/hiai/
# options of compiling APU lib.
WITH_MEDIATEK_APU=OFF
MEDIATEK_APU_SDK_ROOT="$(pwd)/apu_ddk" # Download APU SDK from https://paddlelite-demo.bj.bcebos.com/devices/mediatek/apu_ddk.tar.gz
# options of compiling NNAdapter lib
WITH_NNADAPTER=OFF
NNADAPTER_WITH_HUAWEI_KIRIN_NPU=OFF
NNADAPTER_HUAWEI_KIRIN_NPU_SDK_ROOT="$(pwd)/hiai_ddk_lib_330"
NNADAPTER_WITH_AMLOGIC_NPU=OFF
NNADAPTER_AMLOGIC_NPU_SDK_ROOT="$(pwd)/amlnpu_ddk"
NNADAPTER_WITH_MEDIATEK_APU=OFF
NNADAPTER_MEDIATEK_APU_SDK_ROOT="$(pwd)/apu_ddk" # Download APU SDK from https://paddlelite-demo.bj.bcebos.com/devices/mediatek/apu_ddk.tar.gz
NNADAPTER_WITH_VERISILICON_TIMVX=OFF
NNADAPTER_VERISILICON_TIMVX_SRC_GIT_TAG="main"
NNADAPTER_VERISILICON_TIMVX_VIV_SDK_ROOT=""
NNADAPTER_VERISILICON_TIMVX_VIV_SDK_URL="http://paddlelite-demo.bj.bcebos.com/devices/verisilicon/sdk/viv_sdk_android_9_armeabi_v7a_6_4_4_3_generic.tgz"
NNADAPTER_WITH_ANDROID_NNAPI=OFF
NNADAPTER_WITH_FAKE_DEVICE=OFF
NNADAPTER_FAKE_DEVICE_SDK_ROOT=""
NNADAPTER_WITH_GOOGLE_XNNPACK=OFF
NNADAPTER_GOOGLE_XNNPACK_SRC_GIT_TAG="master"
NNADAPTER_WITH_QUALCOMM_QNN=OFF
NNADAPTER_QUALCOMM_QNN_SDK_ROOT="/usr/local/qnn"
NNADAPTER_QUALCOMM_HEXAGON_SDK_ROOT=""
# options of compiling OPENCL lib.
WITH_OPENCL=OFF
# options of adding training ops
WITH_TRAIN=OFF
# option of time profile, default is OFF
WITH_PROFILE=OFF
# option of precision profile, default is OFF
WITH_PRECISION_PROFILE=OFF
# option of benchmark, default is OFF
WITH_BENCHMARK=OFF
# option of convert_to_ssa_graph
WITH_CONVERT_TO_SSA=ON
# num of threads used during compiling..
readonly NUM_PROC=${LITE_BUILD_THREADS:-4}
#####################################################################################################




#####################################################################################################
# 2. local variables, these variables should not be changed.
#####################################################################################################
# url that stores third-party tar.gz file to accelerate third-party lib installation
readonly THIRDPARTY_URL=https://paddlelite-data.bj.bcebos.com/third_party_libs/
readonly THIRDPARTY_TAR=third-party-801f670.tar.gz
# absolute path of Paddle-Lite.
readonly workspace=$PWD/$(dirname $0)/../../
# basic options for android compiling.
readonly CMAKE_COMMON_OPTIONS="-DWITH_LITE=ON \
                               -DLITE_WITH_ARM=ON \
                               -DLITE_WITH_LIGHT_WEIGHT_FRAMEWORK=ON \
                               -DLITE_WITH_X86=OFF \
                               -DWITH_TESTING=OFF \
                               -DARM_TARGET_OS=android"
# on mac environment, we should expand the maximum file num to compile successfully
os_name=`uname -s`
if [ ${os_name} == "Darwin" ]; then
   ulimit -n 1024
fi

# function of set options for benchmark
function set_benchmark_options {
  WITH_EXTRA=ON
  WITH_EXCEPTION=ON
  BUILD_JAVA=OFF
  WITH_OPENCL=ON
  if [ ${WITH_PROFILE} == "ON" ] || [ ${WITH_PRECISION_PROFILE} == "ON" ]; then
    WITH_LOG=ON
  else
    WITH_LOG=OFF
  fi
}
#####################################################################################################





####################################################################################################
# 3. functions of prepare workspace before compiling
####################################################################################################

# 3.1 generate `__generated_code__.cc`, which is dependended by some targets in cmake.
# here we fake an empty file to make cmake works.
function prepare_workspace {
    local root_dir=$1
    local build_dir=$2
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
####################################################################################################





####################################################################################################
# 4. compiling functions
####################################################################################################

# helper function for setting android api level
function set_android_api_level {
  # android api level for android version
  if [ "${ARCH}" == "armv7" ]; then
      MIN_ANDROID_API_LEVEL=${MIN_ANDROID_API_LEVEL_ARMV7}
  else
      MIN_ANDROID_API_LEVEL=${MIN_ANDROID_API_LEVEL_ARMV8}
  fi
  if [ "${ANDROID_API_LEVEL}" == "Default" ]; then
      cmake_api_level_options=""
  elif [ ${ANDROID_API_LEVEL} -ge ${MIN_ANDROID_API_LEVEL} ]; then
      cmake_api_level_options="-DANDROID_NATIVE_API_LEVEL=${ANDROID_API_LEVEL}"
  else
      echo "Error: ANDROID_API_LEVEL should be no less than ${MIN_ANDROID_API_LEVEL} on ${ARCH}."
      exit 1
  fi
}

# 4.1 function of tiny_publish compiling
# here we only compile light_api lib
function make_tiny_publish_so {

  if [ ! -d third-party ]; then
     git checkout third-party
  fi

  # Step1. Create directory for compiling.
  build_dir=$workspace/build.lite.android.$ARCH.$TOOLCHAIN
  if [ "${WITH_npu}" == "ON" ]; then
      build_dir=${build_dir}.npu
  fi
  if [ -d $build_dir ]; then
      rm -rf $build_dir
  fi
  mkdir -p $build_dir
  cd $build_dir

  # Step2. prepare third-party libs: opencl libs.
  if [ "${WITH_OPENCL}" == "ON" ]; then
      prepare_opencl_source_code $workspace $build_dir
  fi

  # Step3. apply cmake to generate makefiles.
  if [ "${WITH_STRIP}" == "ON" ]; then
      WITH_EXTRA=ON
  fi
  if [ "${BUILD_ARM82_FP16}" == "ON" ]; then
      TOOLCHAIN=clang
  fi

  if [ "$NDK_ROOT" ]; then
      NDK_NAME=$(echo $NDK_ROOT | egrep -o "android-ndk-r[0-9]{2}")
      NDK_VERSION=$(echo $NDK_NAME | egrep -o "[0-9]{2}")
      if [ "$NDK_VERSION" -gt 17 ]; then
          TOOLCHAIN=clang
      fi
  fi

  # android api level for android version
  set_android_api_level

  local cmake_mutable_options="
      -DLITE_BUILD_EXTRA=$WITH_EXTRA \
      -DLITE_WITH_LOG=$WITH_LOG \
      -DLITE_WITH_EXCEPTION=$WITH_EXCEPTION \
      -DLITE_BUILD_TAILOR=$WITH_STRIP \
      -DLITE_OPTMODEL_DIR=$OPTMODEL_DIR \
      -DLITE_WITH_JAVA=$WITH_JAVA \
      -DLITE_WITH_STATIC_LIB=$WITH_STATIC_LIB \
      -DLITE_WITH_CV=$WITH_CV \
      -DLITE_WITH_NPU=$WITH_HUAWEI_KIRIN_NPU \
      -DNPU_DDK_ROOT=$HUAWEI_KIRIN_NPU_SDK_ROOT \
      -DLITE_WITH_APU=$WITH_MEDIATEK_APU \
      -DAPU_DDK_ROOT=$MEDIATEK_APU_SDK_ROOT \
      -DLITE_WITH_NNADAPTER=$WITH_NNADAPTER \
      -DNNADAPTER_WITH_HUAWEI_KIRIN_NPU=$NNADAPTER_WITH_HUAWEI_KIRIN_NPU \
      -DNNADAPTER_HUAWEI_KIRIN_NPU_SDK_ROOT=$NNADAPTER_HUAWEI_KIRIN_NPU_SDK_ROOT \
      -DNNADAPTER_WITH_AMLOGIC_NPU=$NNADAPTER_WITH_AMLOGIC_NPU \
      -DNNADAPTER_AMLOGIC_NPU_SDK_ROOT=$NNADAPTER_AMLOGIC_NPU_SDK_ROOT \
      -DNNADAPTER_WITH_MEDIATEK_APU=$NNADAPTER_WITH_MEDIATEK_APU \
      -DNNADAPTER_MEDIATEK_APU_SDK_ROOT=$NNADAPTER_MEDIATEK_APU_SDK_ROOT \
      -DNNADAPTER_WITH_VERISILICON_TIMVX=$NNADAPTER_WITH_VERISILICON_TIMVX \
      -DNNADAPTER_VERISILICON_TIMVX_SRC_GIT_TAG=$NNADAPTER_VERISILICON_TIMVX_SRC_GIT_TAG \
      -DNNADAPTER_VERISILICON_TIMVX_VIV_SDK_ROOT=$NNADAPTER_VERISILICON_TIMVX_VIV_SDK_ROOT \
      -DNNADAPTER_VERISILICON_TIMVX_VIV_SDK_URL=$NNADAPTER_VERISILICON_TIMVX_VIV_SDK_URL \
      -DNNADAPTER_WITH_ANDROID_NNAPI=$NNADAPTER_WITH_ANDROID_NNAPI \
      -DNNADAPTER_WITH_FAKE_DEVICE=$NNADAPTER_WITH_FAKE_DEVICE \
      -DNNADAPTER_FAKE_DEVICE_SDK_ROOT=$NNADAPTER_FAKE_DEVICE_SDK_ROOT \
      -DNNADAPTER_WITH_GOOGLE_XNNPACK=$NNADAPTER_WITH_GOOGLE_XNNPACK \
      -DNNADAPTER_GOOGLE_XNNPACK_SRC_GIT_TAG=$NNADAPTER_GOOGLE_XNNPACK_SRC_GIT_TAG \
      -DNNADAPTER_WITH_QUALCOMM_QNN=$NNADAPTER_WITH_QUALCOMM_QNN \
      -DNNADAPTER_QUALCOMM_QNN_SDK_ROOT=$NNADAPTER_QUALCOMM_QNN_SDK_ROOT \
      -DNNADAPTER_QUALCOMM_HEXAGON_SDK_ROOT=$NNADAPTER_QUALCOMM_HEXAGON_SDK_ROOT \
      -DLITE_WITH_OPENCL=$WITH_OPENCL \
      -DARM_TARGET_ARCH_ABI=$ARCH \
      -DARM_TARGET_LANG=$TOOLCHAIN \
      -DLITE_WITH_ARM82_FP16=$BUILD_ARM82_FP16 \
      -DLITE_WITH_ARM8_SVE2=$WITH_ARM8_SVE2 \
      -DWITH_ARM_DOTPROD=$WITH_ARM_DOTPROD \
      -DANDROID_STL_TYPE=$ANDROID_STL \
      -DLITE_THREAD_POOL=$WITH_THREAD_POOL \
      -DWITH_CONVERT_TO_SSA=$WITH_CONVERT_TO_SSA"

  cmake $workspace \
      ${CMAKE_COMMON_OPTIONS} \
      ${cmake_api_level_options} \
      ${cmake_mutable_options}  \
      -DLITE_ON_TINY_PUBLISH=ON

  # Step4. Compile libs: cxx_lib, java_lib
  make publish_inference -j$NUM_PROC
  cd - > /dev/null
}

# 4.2 function of full_publish compiling
# here we compile both light_api lib and full_api lib
function make_full_publish_so {

  prepare_thirdparty

  build_directory=$workspace/build.lite.android.$ARCH.$TOOLCHAIN

  if [ -d $build_directory ]
  then
      rm -rf $build_directory
  fi
  mkdir -p $build_directory
  cd $build_directory

  prepare_workspace $workspace $build_directory

  if [ "${WITH_BENCHMARK}" == "ON" ]; then
      set_benchmark_options
  fi

  if [ "${WITH_OPENCL}" == "ON" ]; then
      prepare_opencl_source_code $workspace $build_dir
  fi

  if [ "${WITH_STRIP}" == "ON" ]; then
      WITH_EXTRA=ON
  fi
  if [ "${BUILD_ARM82_FP16}" == "ON" ]; then
      TOOLCHAIN=clang
  fi

  if [ "$NDK_ROOT" ]; then
      NDK_NAME=$(echo $NDK_ROOT | egrep -o "android-ndk-r[0-9]{2}")
      NDK_VERSION=$(echo $NDK_NAME | egrep -o "[0-9]{2}")
      if [ "$NDK_VERSION" -gt 17 ]; then
          TOOLCHAIN=clang
      fi
  fi

  # android api level for android version
  set_android_api_level

  local cmake_mutable_options="
      -DLITE_BUILD_EXTRA=$WITH_EXTRA \
      -DLITE_WITH_LOG=$WITH_LOG \
      -DLITE_WITH_EXCEPTION=$WITH_EXCEPTION \
      -DLITE_BUILD_TAILOR=$WITH_STRIP \
      -DLITE_OPTMODEL_DIR=$OPTMODEL_DIR \
      -DLITE_WITH_JAVA=$WITH_JAVA \
      -DLITE_WITH_STATIC_LIB=$WITH_STATIC_LIB \
      -DLITE_WITH_CV=$WITH_CV \
      -DLITE_WITH_NPU=$WITH_HUAWEI_KIRIN_NPU \
      -DNPU_DDK_ROOT=$HUAWEI_KIRIN_NPU_SDK_ROOT \
      -DLITE_WITH_APU=$WITH_MEDIATEK_APU \
      -DAPU_DDK_ROOT=$MEDIATEK_APU_SDK_ROOT \
      -DLITE_WITH_NNADAPTER=$WITH_NNADAPTER \
      -DNNADAPTER_WITH_HUAWEI_KIRIN_NPU=$NNADAPTER_WITH_HUAWEI_KIRIN_NPU \
      -DNNADAPTER_HUAWEI_KIRIN_NPU_SDK_ROOT=$NNADAPTER_HUAWEI_KIRIN_NPU_SDK_ROOT \
      -DNNADAPTER_WITH_AMLOGIC_NPU=$NNADAPTER_WITH_AMLOGIC_NPU \
      -DNNADAPTER_AMLOGIC_NPU_SDK_ROOT=$NNADAPTER_AMLOGIC_NPU_SDK_ROOT \
      -DNNADAPTER_WITH_MEDIATEK_APU=$NNADAPTER_WITH_MEDIATEK_APU \
      -DNNADAPTER_MEDIATEK_APU_SDK_ROOT=$NNADAPTER_MEDIATEK_APU_SDK_ROOT \
      -DNNADAPTER_WITH_VERISILICON_TIMVX=$NNADAPTER_WITH_VERISILICON_TIMVX \
      -DNNADAPTER_VERISILICON_TIMVX_SRC_GIT_TAG=$NNADAPTER_VERISILICON_TIMVX_SRC_GIT_TAG \
      -DNNADAPTER_VERISILICON_TIMVX_VIV_SDK_ROOT=$NNADAPTER_VERISILICON_TIMVX_VIV_SDK_ROOT \
      -DNNADAPTER_VERISILICON_TIMVX_VIV_SDK_URL=$NNADAPTER_VERISILICON_TIMVX_VIV_SDK_URL \
      -DNNADAPTER_WITH_ANDROID_NNAPI=$NNADAPTER_WITH_ANDROID_NNAPI \
      -DNNADAPTER_WITH_FAKE_DEVICE=$NNADAPTER_WITH_FAKE_DEVICE \
      -DNNADAPTER_FAKE_DEVICE_SDK_ROOT=$NNADAPTER_FAKE_DEVICE_SDK_ROOT \
      -DNNADAPTER_WITH_GOOGLE_XNNPACK=$NNADAPTER_WITH_GOOGLE_XNNPACK \
      -DNNADAPTER_GOOGLE_XNNPACK_SRC_GIT_TAG=$NNADAPTER_GOOGLE_XNNPACK_SRC_GIT_TAG \
      -DNNADAPTER_WITH_QUALCOMM_QNN=$NNADAPTER_WITH_QUALCOMM_QNN \
      -DNNADAPTER_QUALCOMM_QNN_SDK_ROOT=$NNADAPTER_QUALCOMM_QNN_SDK_ROOT \
      -DNNADAPTER_QUALCOMM_HEXAGON_SDK_ROOT=$NNADAPTER_QUALCOMM_HEXAGON_SDK_ROOT \
      -DLITE_WITH_OPENCL=$WITH_OPENCL \
      -DARM_TARGET_ARCH_ABI=$ARCH \
      -DARM_TARGET_LANG=$TOOLCHAIN \
      -DLITE_WITH_TRAIN=$WITH_TRAIN \
      -DLITE_WITH_PROFILE=$WITH_PROFILE \
      -DLITE_WITH_ARM82_FP16=$BUILD_ARM82_FP16 \
      -DLITE_WITH_ARM8_SVE2=$WITH_ARM8_SVE2 \
      -DWITH_ARM_DOTPROD=$WITH_ARM_DOTPROD \
      -DLITE_WITH_PRECISION_PROFILE=$WITH_PRECISION_PROFILE \
      -DANDROID_STL_TYPE=$ANDROID_STL \
      -DWITH_CONVERT_TO_SSA=$WITH_CONVERT_TO_SSA"

  cmake $workspace \
      ${CMAKE_COMMON_OPTIONS} \
      ${cmake_api_level_options} \
      ${cmake_mutable_options}

  if [ "${WITH_BENCHMARK}" == "ON" ]; then
    make benchmark_bin -j$NUM_PROC
  else
    make publish_inference -j$NUM_PROC
  fi
  cd - > /dev/null
}


# 4.3 function of print help information
function print_usage {
    echo "----------------------------------------------------------------------------------------------------------------------------------------"
    echo -e "| Methods of compiling Padddle-Lite Android library:                                                                                   |"
    echo "----------------------------------------------------------------------------------------------------------------------------------------"
    echo -e "|  compile android library: (armv8, gcc, c++_static)                                                                                   |"
    echo -e "|     ./lite/tools/build_android.sh                                                                                                    |"
    echo -e "|  print help information:                                                                                                             |"
    echo -e "|     ./lite/tools/build_android.sh help                                                                                               |"
    echo -e "|                                                                                                                                      |"
    echo -e "|  optional argument:                                                                                                                  |"
    echo -e "|     --arch: (armv8|armv7), default is armv8                                                                                          |"
    echo -e "|     --toolchain: (gcc|clang), default is gcc                                                                                         |"
    echo -e "|     --android_stl: (c++_static|c++_shared), default is c++_static                                                                    |"
    echo -e "|     --with_java: (OFF|ON); controls whether to publish java api lib, default is ON                                                   |"
    echo -e "|     --with_static_lib: (OFF|ON); controls whether to publish c++ api static lib, default is OFF                                      |"
    echo -e "|     --with_cv: (OFF|ON); controls whether to compile cv functions into lib, default is OFF                                           |"
    echo -e "|     --with_log: (OFF|ON); controls whether to print log information, default is ON                                                   |"
    echo -e "|     --with_convert_to_ssa: (OFF|ON); controls whether to modify input model graph which is not DAG to SSA graph, default is OFF      |"
    echo -e "|     --with_exception: (OFF|ON); controls whether to throw the exception when error occurs, default is OFF                            |"
    echo -e "|     --with_extra: (OFF|ON); controls whether to publish extra operators and kernels for (sequence-related model such as OCR or NLP)  |"
    echo -e "|     --with_profile: (OFF|ON); controls whether to support time profile, default is OFF                                               |"
    echo -e "|     --with_precision_profile: (OFF|ON); controls whether to support precision profile, default is OFF                                |"
    echo -e "|     --with_arm82_fp16: (OFF|ON); controls whether to include FP16 kernels, default is OFF                                            |"
    echo -e "|                                  warning: when --with_arm82_fp16=ON, toolchain will be set as clang, arch will be set as armv8.      |"
    echo -e "|     --with_arm8_sve2: (OFF|ON); controls whether to include SVE2 kernels, default is OFF                                             |"
    echo -e "|                                  warning: when --with_arm8_sve2=ON, NDK version need >= r23, arch will be set as armv8.              |"
    echo -e "|     --android_api_level: (16~27); control android api level, default is 16 on armv7 and 21 on armv8. You could set a specific        |"
    echo -e "|             android_api_level as you need.                                                                                           |"
    echo -e "|                       | Paddle-Lite Requird / ARM ABI      | armv7 | armv8 |                                                         |"
    echo -e "|                       |------------------------------------|-------|-------|                                                         |"
    echo -e "|                       |Supported Minimum Android API Level |  16   |  21   |                                                         |"
    echo -e "|                       |Supported Minimum Android Version   |  4.1  |  5.0  |                                                         |"
    echo -e "|     --with_benchmark: (OFF|ON); controls whether to compile benchmark binary, default is OFF                                         |"
    echo -e "|                                                                                                                                      |"
    echo -e "|  arguments of benchmark binary compiling:(armv8, gcc, c++_static)                                                                    |"
    echo -e "|     ./lite/tools/build_android.sh --with_benchmark=ON full_publish                                                                   |"
    echo -e "|                                                                                                                                      |"
    echo -e "|  arguments of striping lib according to input model:(armv8, gcc, c++_static)                                                         |"
    echo -e "|     ./lite/tools/build_android.sh --with_strip=ON --opt_model_dir=YourOptimizedModelDir                                              |"
    echo -e "|     --with_strip: (OFF|ON); controls whether to strip lib accrding to input model, default is OFF                                    |"
    echo -e "|     --opt_model_dir: (absolute path to optimized model dir) required when compiling striped library                                  |"
    echo -e "|  detailed information about striping lib:  https://paddle-lite.readthedocs.io/zh/latest/user_guides/library_tailoring.html           |"
    echo -e "|                                                                                                                                      |"
    echo -e "|  arguments of npu library compiling:(armv8, gcc, c++_static)                                                                         |"
    echo -e "|     ./lite/tools/build_android.sh --with_huawei_kirin_npu=ON --huawei_kirin_npu_sdk_root=YourNpuSdkPath                              |"
    echo -e "|     --with_huawei_kirin_npu: (OFF|ON); controls whether to compile lib for huawei_kirin_npu, default is OFF                          |"
    echo -e "|     --huawei_kirin_npu_sdk_root: (path to huawei HiAi DDK file) required when compiling npu library                                  |"
    echo -e "|             you can download huawei HiAi DDK from:  https://developer.huawei.com/consumer/cn/hiai/                                   |"
    echo -e "|  detailed information about Paddle-Lite NPU:  https://paddle-lite.readthedocs.io/zh/latest/demo_guides/npu.html                      |"
    echo -e "|                                                                                                                                      |"
    echo -e "|  arguments of apu library compiling:(armv8, gcc, c++_static)                                                                         |"
    echo -e "|     ./lite/tools/build_android.sh --with_mediatek_apu=ON --mediatek_apu_sdk_root=YourApuSdkPath                                      |"
    echo -e "|     --with_mediatek_apu: (OFF|ON); controls whether to compile lib for mediatek_apu, default is OFF                                  |"
    echo -e "|     --mediatek_apu_sdk_root: (path to mediatek APU SDK file) required when compiling apu library                                     |"
    echo -e "|             you can download mediatek APU SDK from:  https://paddlelite-demo.bj.bcebos.com/devices/mediatek/apu_ddk.tar.gz           |"
    echo -e "|  detailed information about Paddle-Lite APU:  https://paddle-lite.readthedocs.io/zh/latest/demo_guides/mediatek_apu.html             |"
    echo -e "|                                                                                                                                      |"
    echo -e "|  arguments of opencl library compiling:(armv8, gcc, c++_static)                                                                      |"
    echo -e "|     ./lite/tools/build_android.sh --with_opencl=ON                                                                                   |"
    echo -e "|     --with_opencl: (OFF|ON); controls whether to compile lib for opencl, default is OFF                                              |"
    echo "----------------------------------------------------------------------------------------------------------------------------------------"
    echo
}

####################################################################################################


####################################################################################################
# 5. main functions: choose compiling method according to input argument
####################################################################################################
function main {
    if [ -z "$1" ]; then
        # compiling result contains light_api lib only, recommanded.
        make_tiny_publish_so $ARCH $TOOLCHAIN $ANDROID_STL
        exit 0
    fi

    # Parse command line.
    for i in "$@"; do
        case $i in
            # armv7 or armv8, default armv8
            --arch=*)
                ARCH="${i#*=}"
                shift
                ;;
            # gcc or clang, default gcc
            --toolchain=*)
                TOOLCHAIN="${i#*=}"
                shift
                ;;
            # c++_static or c++_shared, default c++_static
            --android_stl=*)
                ANDROID_STL="${i#*=}"
                shift
                ;;
            --android_api_level=*)
                ANDROID_API_LEVEL="${i#*=}"
                shift
                ;;
            # ON or OFF, default OFF
            --with_extra=*)
                WITH_EXTRA="${i#*=}"
                shift
                ;;
            # ON or OFF, default OFF
            --with_cv=*)
                WITH_CV="${i#*=}"
                shift
                ;;
            # ON or OFF, default ON
            --with_java=*)
                WITH_JAVA="${i#*=}"
                shift
                ;;
            # ON or OFF, default OFF
            --with_strip=*)
                WITH_STRIP="${i#*=}"
                shift
                ;;
            # ON or OFF, default OFF
            --with_thread_pool=*)
                WITH_THREAD_POOL="${i#*=}"
                shift
                ;;
            # string, absolute path to optimized model dir
            --opt_model_dir=*)
                OPTMODEL_DIR="${i#*=}"
                shift
                ;;
            # ON or OFF, default ON
            --with_log=*)
                WITH_LOG="${i#*=}"
                shift
                ;;
            # ON or OFF, default OFF
            --with_exception=*)
                WITH_EXCEPTION="${i#*=}"
                if [[ $WITH_EXCEPTION == "ON" && $ARCH == "armv7" && $TOOLCHAIN != "clang" ]]; then
                     set +x
                     echo
                     echo -e "Error: only clang provide C++ exception handling support for 32-bit ARM."
                     echo
                     exit 1
                fi
                shift
                ;;
            # compiling lib which can operate on opencl and cpu.
            --with_opencl=*)
                WITH_OPENCL="${i#*=}"
                shift
                ;;
            # compiling lib which can operate on huawei npu.
            --with_huawei_kirin_npu=*)
                WITH_HUAWEI_KIRIN_NPU="${i#*=}"
                shift
                ;;
            --huawei_kirin_npu_sdk_root=*)
                HUAWEI_KIRIN_NPU_SDK_ROOT="${i#*=}"
                shift
                ;;
            # compiling lib which can operate on mediatek apu.
            --with_mediatek_apu=*)
                WITH_MEDIATEK_APU="${i#*=}"
                shift
                ;;
            --mediatek_apu_sdk_root=*)
                MEDIATEK_APU_SDK_ROOT="${i#*=}"
                shift
                ;;
            # compiling lib which can operate on nnadapter.
            --with_nnadapter=*)
                WITH_NNADAPTER="${i#*=}"
                shift
                ;;
            --nnadapter_with_huawei_kirin_npu=*)
                NNADAPTER_WITH_HUAWEI_KIRIN_NPU="${i#*=}"
                shift
                ;;
            --nnadapter_huawei_kirin_npu_sdk_root=*)
                NNADAPTER_HUAWEI_KIRIN_NPU_SDK_ROOT="${i#*=}"
                shift
                ;;
            --nnadapter_with_amlogic_npu=*)
                NNADAPTER_WITH_AMLOGIC_NPU="${i#*=}"
                shift
                ;;
            --nnadapter_amlogic_npu_sdk_root=*)
                NNADAPTER_AMLOGIC_NPU_SDK_ROOT="${i#*=}"
                shift
                ;;
            --nnadapter_with_mediatek_apu=*)
                NNADAPTER_WITH_MEDIATEK_APU="${i#*=}"
                shift
                ;;
            --nnadapter_mediatek_apu_sdk_root=*)
                NNADAPTER_MEDIATEK_APU_SDK_ROOT="${i#*=}"
                shift
                ;;
            --nnadapter_with_verisilicon_timvx=*)
                NNADAPTER_WITH_VERISILICON_TIMVX="${i#*=}"
                shift
                ;;
            --nnadapter_verisilicon_timvx_src_git_tag=*)
                NNADAPTER_VERISILICON_TIMVX_SRC_GIT_TAG="${i#*=}"
                shift
                ;;
            --nnadapter_verisilicon_timvx_viv_sdk_root=*)
                NNADAPTER_VERISILICON_TIMVX_VIV_SDK_ROOT="${i#*=}"
                shift
                ;;
            --nnadapter_verisilicon_timvx_viv_sdk_url=*)
                NNADAPTER_VERISILICON_TIMVX_VIV_SDK_URL="${i#*=}"
                shift
                ;;
            --nnadapter_with_android_nnapi=*)
                NNADAPTER_WITH_ANDROID_NNAPI="${i#*=}"
                shift
                ;;
            --nnadapter_with_fake_device=*)
                NNADAPTER_WITH_FAKE_DEVICE="${i#*=}"
                shift
                ;;
            --nnadapter_fake_device_sdk_root=*)
                NNADAPTER_FAKE_DEVICE_SDK_ROOT="${i#*=}"
                shift
                ;;
            --nnadapter_with_google_xnnpack=*)
                NNADAPTER_WITH_GOOGLE_XNNPACK="${i#*=}"
                shift
                ;;
            --nnadapter_google_xnnpack_src_git_tag=*)
                NNADAPTER_GOOGLE_XNNPACK_SRC_GIT_TAG="${i#*=}"
                shift
                ;;
            --nnadapter_with_qualcomm_qnn=*)
                NNADAPTER_WITH_QUALCOMM_QNN="${i#*=}"
                shift
                ;;
            --nnadapter_qualcomm_qnn_sdk_root=*)
                NNADAPTER_QUALCOMM_QNN_SDK_ROOT="${i#*=}"
                shift
                ;;
            --nnadapter_qualcomm_hexagon_sdk_root=*)
                NNADAPTER_QUALCOMM_HEXAGON_SDK_ROOT="${i#*=}"
                shift
                ;;
            # compiling result contains both light_api and cxx_api lib.
            full_publish)
                make_full_publish_so
                exit 0
                ;;
            # compiling lib with training ops.
            --with_train=*)
                WITH_TRAIN="${i#*=}"
                shift
                ;;
            # compiling lib with time profile, default OFF.
            --with_profile=*)
                WITH_PROFILE="${i#*=}"
                shift
                ;;
            # compiling lib with precision profile, default OFF.
            --with_precision_profile=*)
                WITH_PRECISION_PROFILE="${i#*=}"
                shift
                ;;
            # compiling lib with benchmark feature, default OFF.
            --with_benchmark=*)
                WITH_BENCHMARK="${i#*=}"
                shift
                ;;
            # controls whether to include FP16 kernels, default is OFF
            --with_arm82_fp16=*)
                BUILD_ARM82_FP16="${i#*=}"
                shift
                ;;
            # controls whether to compile cplus static library, default is OFF
            --with_static_lib=*)
                WITH_STATIC_LIB="${i#*=}"
                shift
                ;;
            --with_convert_to_ssa=*)
                WITH_CONVERT_TO_SSA="${i#*=}"
                shift
                ;;
            --with_arm8_sve2=*)
                WITH_ARM8_SVE2="${i#*=}"
                shift
                ;;
            --with_arm_dotprod=*)
                WITH_ARM_DOTPROD="${i#*=}"
                shift
                ;;
            help)
            # print help info
                print_usage
                exit 0
                ;;
            *)
                # unknown option
                echo "Error: unsupported argument \"${i#*=}\""
                print_usage
                exit 1
                ;;
        esac
    done
    # compiling result contains light_api lib only, recommanded.
    make_tiny_publish_so
    exit 0
}

main $@
