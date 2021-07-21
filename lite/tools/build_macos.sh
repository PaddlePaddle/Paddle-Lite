#!/bin/bash
set +x
set -e

#####################################################################################################
# 1. global variables, you can change them according to your requirements
#####################################################################################################
# only support armv8|x86.
ARCH=armv8
# ON or OFF, default OFF.

CMAKE_EXTRA_OPTIONS=""
BUILD_EXTRA=OFF
BUILD_TRAIN=OFF
BUILD_JAVA=ON
BUILD_PYTHON=OFF
BUILD_DIR=$(pwd)
OPTMODEL_DIR=""
BUILD_TAILOR=OFF
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
BUILD_NPU=OFF
NPU_DDK_ROOT="$(pwd)/ai_ddk_lib/" # Download HiAI DDK from https://developer.huawei.com/consumer/cn/hiai/
BUILD_XPU=OFF
BUILD_XTCL=OFF
XPU_SDK_ROOT=""
XPU_SDK_URL=""
XPU_SDK_ENV=""
BUILD_APU=OFF
APU_DDK_ROOT="$(pwd)/apu_sdk_lib/"
BUILD_RKNPU=OFF
RKNPU_DDK_ROOT="$(pwd)/rknpu/"
WITH_HUAWEI_ASCEND_NPU=OFF # Huawei Ascend Builder/Runtime Libs on X86 host 
# default installation path, ensure acllib/atc/opp directories are all in this root dir
HUAWEI_ASCEND_NPU_DDK_ROOT="/usr/local/Ascend/ascend-toolkit/latest/x86_64-linux_gcc4.8.5"
PYTHON_EXECUTABLE_OPTION=""
workspace=$PWD/$(dirname $0)/../../
OPTMODEL_DIR=""
IOS_DEPLOYMENT_TARGET=11.0
# num of threads used during compiling..
readonly NUM_PROC=${LITE_BUILD_THREADS:-4}
#####################################################################################################


#####################################################################################################
# 2. local variables, these variables should not be changed.
#####################################################################################################
# on mac environment, we should expand the maximum file num to compile successfully
os_name=`uname -s`
if [ ${os_name} == "Darwin" ]; then
   ulimit -n 1024
fi
#####################################################################################################

####################################################################################################
# 3. compiling functions
####################################################################################################
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
    if [ ! -d $workspace/third-party -o -f $workspace/third-party-ea5576.tar.gz ]; then
        rm -rf $workspace/third-party

        if [ ! -f $workspace/third-party-ea5576.tar.gz ]; then
            wget $THIRDPARTY_TAR
        fi
        tar xzf third-party-ea5576.tar.gz
    else
        git submodule update --init --recursive
    fi
}

function make_armosx {
    local arch=armv8
    local os=armmacos
       if [ "${WITH_STRIP}" == "ON" ]; then
        WITH_EXTRA=ON
    fi

    build_dir=$workspace/build.macos.${os}.${arch}
    if [ -d $build_dir ]
    then
        rm -rf $build_dir
    fi
    echo "building arm macos target into $build_dir"
    echo "target arch: $arch"
    mkdir -p ${build_dir}
    cd ${build_dir}
    GEN_CODE_PATH_PREFIX=lite/gen_code
    mkdir -p ./${GEN_CODE_PATH_PREFIX}
    touch ./${GEN_CODE_PATH_PREFIX}/__generated_code__.cc
    cmake $workspace \
            -DWITH_LITE=ON \
            -DLITE_WITH_ARM=ON \
            -DLITE_ON_TINY_PUBLISH=ON \
            -DLITE_WITH_OPENMP=OFF \
            -DWITH_ARM_DOTPROD=OFF \
            -DLITE_WITH_LIGHT_WEIGHT_FRAMEWORK=ON \
            -DLITE_WITH_X86=OFF \
            -DLITE_WITH_LOG=$WITH_LOG \
            -DLITE_WITH_EXCEPTION=$WITH_EXCEPTION \
            -DLITE_BUILD_TAILOR=$BUILD_TAILOR \
            -DLITE_OPTMODEL_DIR=$OPTMODEL_DIR \
            -DARM_TARGET_ARCH_ABI=$arch \
            -DLITE_BUILD_EXTRA=$BUILD_EXTRA \
            -DLITE_WITH_CV=$BUILD_CV \
            -DDEPLOYMENT_TARGET=${IOS_DEPLOYMENT_TARGET} \
            -DARM_TARGET_OS=armmacos

    make publish_inference -j$NUM_PROC
    cd -
}

function make_x86 {
  prepare_thirdparty

  root_dir=$(pwd)
  build_directory=$BUILD_DIR/build.lite.x86

  if [ ${WITH_HUAWEI_ASCEND_NPU} == "ON" ]; then
    export CXX=g++ # Huawei Ascend NPU need g++
    build_directory=$BUILD_DIR/build.lite.huawei_ascend_npu
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
            -DWITH_TESTING=OFF \
            -DWITH_AVX=${WITH_AVX} \
            -DWITH_MKLDNN=OFF    \
            -DLITE_WITH_X86=ON  \
            -DWITH_LITE=ON \
            -DLITE_WITH_LIGHT_WEIGHT_FRAMEWORK=OFF \
            -DLITE_WITH_ARM=OFF \
            -DLITE_WITH_OPENCL=${WITH_OPENCL} \
            -DWITH_GPU=OFF \
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
            -DLITE_WITH_HUAWEI_ASCEND_NPU=$WITH_HUAWEI_ASCEND_NPU \
            -DHUAWEI_ASCEND_NPU_DDK_ROOT=$HUAWEI_ASCEND_NPU_DDK_ROOT \
            -DCMAKE_BUILD_TYPE=Release \
            -DPY_VERSION=$PY_VERSION \
            $PYTHON_EXECUTABLE_OPTION

  make publish_inference -j$NUM_PROC
  cd -
}

function print_usage {
    echo "----------------------------------------------------------------------------------------------------------------------------------------"
    echo -e "| Methods of compiling Padddle-Lite Arm Macos library:                                                                                 |"
    echo "----------------------------------------------------------------------------------------------------------------------------------------"
    echo -e "|  compile macos armv8 library:                                                                                                        |"
    echo -e "|     ./lite/tools/build_macos.sh                                                                                                      |"
    echo -e "|  print help information:                                                                                                             |"
    echo -e "|     ./lite/tools/build_macos.sh help                                                                                                 |"
    echo -e "|                                                                                                                                      |"
    echo -e "|  for arm macos:                                                                                                                      |"
    echo -e "|  optional argument:                                                                                                                  |"
    echo -e "|     --arch: arm macos only support armv8                                                                                             |"
    echo -e "|     --with_cv: (OFF|ON); controls whether to compile cv functions into lib, default is OFF                                           |"
    echo -e "|     --with_log: (OFF|ON); controls whether to print log information, default is ON                                                   |"
    echo -e "|     --with_exception: (OFF|ON); controls whether to throw the exception when error occurs, default is OFF                            |"
    echo -e "|     --with_extra: (OFF|ON); controls whether to publish extra operators and kernels for (sequence-related model such as OCR or NLP)  |"
    echo -e "|                                                                                                                                      |"
    echo -e "|  arguments of striping lib according to input model:(armv8, gcc, c++_static)                                                         |"
    echo -e "|     ./lite/tools/build_macos.sh --with_strip=ON --opt_model_dir=YourOptimizedModelDir                                                |"
    echo -e "|     --with_strip: (OFF|ON); controls whether to strip lib accrding to input model, default is OFF                                    |"
    echo -e "|     --opt_model_dir: (absolute path to optimized model dir) required when compiling striped library                                  |"
    echo -e "|  detailed information about striping lib:  https://paddle-lite.readthedocs.io/zh/latest/user_guides/library_tailoring.html           |"
    echo "----------------------------------------------------------------------------------------------------------------------------------------"

}

function main {
    if [ -z "$1" ]; then
        make_osx $ARCH
        exit 0
    fi

    # Parse command line.
    for i in "$@"; do
        case $i in
            --build_extra=*)
                BUILD_EXTRA="${i#*=}"
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
            --build_opencl=*)
                WITH_OPENCL="${i#*=}"
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
            --with_huawei_ascend_npu=*)
                WITH_HUAWEI_ASCEND_NPU="${i#*=}"
                shift
                ;;
            --huawei_ascend_npu_ddk_root=*)
                HUAWEI_ASCEND_NPU_DDK_ROOT="${i#*=}"
                shift
                ;;
            --ios_deployment_target=*)
                IOS_DEPLOYMENT_TARGET="${i#*=}"
                shift
                ;;
            arm64)
               make_armosx
               shift
               ;;
            x86)
               make_x86
               shift
               ;;
           help)
                print_usage
                exit 0
                ;;
            *)
                # unknown option
                print_usage
                exit 1
                ;;
        esac
    done
    exit 0
}

main $@
