#!/bin/bash
set -e

#####################################################################################################
# 1. global variables, you can change them according to your requirements
#####################################################################################################
# armv8 or armv7hf or armv7 or x86, default armv8.
ARCH=armv8
# gcc or clang, default gcc.
TOOLCHAIN=gcc
# ON or OFF, default OFF.
WITH_EXTRA=OFF
# controls whether to compile python lib, default is OFF.
WITH_PYTHON=OFF
PY_VERSION=""

# ON or OFF, default is OFF
WITH_STATIC_LIB=OFF
# controls whether to compile cv functions into lib, default is OFF.
WITH_CV=OFF
# controls whether to print log information, default is ON.
WITH_LOG=ON
# controls whether to throw the exception when error occurs, default is OFF
WITH_EXCEPTION=OFF
# options of striping lib according to input model.
WITH_STRIP=OFF
OPTMODEL_DIR=""
# options of compiling OPENCL lib.
WITH_OPENCL=OFF
# options of compiling NNAdapter lib
WITH_NNADAPTER=OFF
NNADAPTER_WITH_QUALCOMM_QNN=OFF
NNADAPTER_QUALCOMM_QNN_SDK_ROOT="/usr/local/qnn"
NNADAPTER_QUALCOMM_HEXAGON_SDK_ROOT=
NNADAPTER_WITH_FAKE_DEVICE=OFF
NNADAPTER_FAKE_DEVICE_SDK_ROOT=""

# options of adding training ops
WITH_TRAIN=OFF
# options of building tiny publish so
WITH_TINY_PUBLISH=ON
# controls whether to include FP16 kernels, default is OFF
BUILD_ARM82_FP16=OFF
# options of profiling
WITH_PROFILE=OFF
WITH_PRECISION_PROFILE=OFF
# option of benchmark, default is OFF
WITH_BENCHMARK=OFF
# option of light weight framework, default is OFF
WITH_LIGHT_WEIGHT_FRAMEWORK=OFF
# num of threads used during compiling..
readonly NUM_PROC=${LITE_BUILD_THREADS:-4}
#####################################################################################################


#####################################################################################################
# 2. local variables, these variables should not be changed.
#####################################################################################################
# url that stores third-party tar.gz file to accelerate third-party lib installation
readonly THIRDPARTY_URL=https://paddlelite-data.bj.bcebos.com/third_party_libs/
readonly THIRDPARTY_TAR=third-party-91a9ab3.tar.gz

# absolute path of Paddle-Lite.
readonly workspace=$PWD/$(dirname $0)/../../
# basic options for linux compiling.
readonly CMAKE_COMMON_OPTIONS="-DWITH_LITE=ON \
                            -DCMAKE_BUILD_TYPE=Release \
                            -DWITH_MKL=OFF \
                            -DWITH_MKLDNN=OFF \
                            -DWITH_TESTING=OFF \
                            -DLITE_WITH_OPENMP=OFF"

# function of set options for benchmark
function set_benchmark_options {
  WITH_EXTRA=ON
  WITH_EXCEPTION=ON
  WITH_NNADAPTER=ON
  if [ "${ARCH}" == "x86" ]; then
    # Turn off opencl. Additional third party library need to be installed on
    # Linux. Otherwise opencl is not supported on Linux. See link for more info:
    # https://software.intel.com/content/www/us/en/develop/articles/opencl-drivers.html
    WITH_OPENCL=OFF
    WITH_LIGHT_WEIGHT_FRAMEWORK=OFF
  else
    WITH_LIGHT_WEIGHT_FRAMEWORK=ON
    WITH_OPENCL=ON
  fi
  if [ ${WITH_PROFILE} == "ON" ] || [ ${WITH_PRECISION_PROFILE} == "ON" ]; then
    WITH_LOG=ON
  else
    WITH_LOG=OFF
  fi
}

# mutable options for linux compiling.
function init_cmake_mutable_options {
    if [ "$WITH_PYTHON" = "ON" -a "$WITH_TINY_PUBLISH" = "ON" ]; then
        echo "Warning: build full_publish to use python."
        WITH_TINY_PUBLISH=OFF
    fi
    if [ "$WITH_TRAIN" = "ON" -a "$WITH_TINY_PUBLISH" = "ON" ]; then
        echo "Warning: build full_publish to add training ops."
        WITH_TINY_PUBLISH=OFF
    fi

    if [ "$BUILD_TAILOR" = "ON" -a "$OPTMODEL_DIR" = "" ]; then
        echo "Error: set OPTMODEL_DIR if BUILD_TAILOR is ON."
        exit 1
    fi

    arm_arch=$ARCH
    arm_target_os=qnx
    WITH_LIGHT_WEIGHT_FRAMEWORK=ON
    WITH_AVX=OFF

    if [ "${WITH_STRIP}" == "ON" ]; then
        WITH_EXTRA=ON
    fi

    if [ "${WITH_BENCHMARK}" == "ON" ]; then
        set_benchmark_options
    fi

    cmake_mutable_options="-DLITE_WITH_ARM=ON \
                        -DLITE_WITH_X86=OFF \
                        -DARM_TARGET_ARCH_ABI=$arm_arch \
                        -DARM_TARGET_OS=$arm_target_os \
                        -DARM_TARGET_LANG=$TOOLCHAIN \
                        -DLITE_WITH_LIGHT_WEIGHT_FRAMEWORK=$WITH_LIGHT_WEIGHT_FRAMEWORK \
                        -DLITE_BUILD_EXTRA=$WITH_EXTRA \
                        -DLITE_WITH_PYTHON=$WITH_PYTHON \
                        -DPY_VERSION=$PY_VERSION \
                        -DLITE_WITH_STATIC_LIB=$WITH_STATIC_LIB \
                        -DLITE_WITH_CV=$WITH_CV \
                        -DLITE_WITH_LOG=$WITH_LOG \
                        -DLITE_WITH_EXCEPTION=$WITH_EXCEPTION \
                        -DLITE_BUILD_TAILOR=$WITH_STRIP \
                        -DLITE_OPTMODEL_DIR=$OPTMODEL_DIR \
                        -DWITH_STATIC_MKL=OFF \
                        -DWITH_AVX=OFF \
                        -DLITE_WITH_OPENCL=$WITH_OPENCL \
                        -DLITE_WITH_TRAIN=$WITH_TRAIN  \
                        -DLITE_WITH_NNADAPTER=$WITH_NNADAPTER \
                        -DNNADAPTER_WITH_QUALCOMM_QNN=$NNADAPTER_WITH_QUALCOMM_QNN \
                        -DNNADAPTER_QUALCOMM_QNN_SDK_ROOT=$NNADAPTER_QUALCOMM_QNN_SDK_ROOT \
                        -DNNADAPTER_QUALCOMM_HEXAGON_SDK_ROOT=$NNADAPTER_QUALCOMM_HEXAGON_SDK_ROOT \
                        -DNNADAPTER_WITH_FAKE_DEVICE=$NNADAPTER_WITH_FAKE_DEVICE \
                        -DNNADAPTER_FAKE_DEVICE_SDK_ROOT=$NNADAPTER_FAKE_DEVICE_SDK_ROOT \
                        -DLITE_WITH_PROFILE=${WITH_PROFILE} \
                        -DLITE_WITH_ARM82_FP16=$BUILD_ARM82_FP16 \
                        -DLITE_WITH_PRECISION_PROFILE=${WITH_PRECISION_PROFILE} \
                        -DLITE_ON_TINY_PUBLISH=$WITH_TINY_PUBLISH"

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

# 4.1 function of publish compiling
# here we only compile light_api lib
function make_publish_so {
    init_cmake_mutable_options

    if [ "$WITH_TINY_PUBLISH" = "OFF" ]; then
        prepare_thirdparty
    else
        if [ ! -d third-party ] ; then
            git checkout third-party
        fi
    fi

    build_dir=$workspace/build.lite.qnx.$ARCH.$TOOLCHAIN
    if [ "${WITH_OPENCL}" = "ON" ]; then
        build_dir=${build_dir}.opencl
    fi

    if [ -d $build_dir ]; then
        rm -rf $build_dir
    fi
    mkdir -p $build_dir
    cd $build_dir

    prepare_workspace $workspace $build_dir

    if [ "${WITH_OPENCL}" = "ON" ]; then
        prepare_opencl_source_code $workspace $build_dir
    fi

    cmake $workspace \
        ${CMAKE_COMMON_OPTIONS} \
        ${cmake_mutable_options}

    if [ "${WITH_BENCHMARK}" == "ON" ]; then
        make benchmark_bin -j$NUM_PROC
    else
        make publish_inference -j$NUM_PROC
    fi
    cd - > /dev/null
}
####################################################################################################



function print_usage {
    echo "--------------------------------------------------------------------------------------------------------------------------------------------------------"
    echo -e "| Methods of compiling Padddle-Lite Linux library:                                                                                                     |"
    echo "--------------------------------------------------------------------------------------------------------------------------------------------------------"
    echo -e "|  compile linux library: (armv8, gcc)                                                                                                                 |"
    echo -e "|     ./lite/tools/build_linux.sh                                                                                                                      |"
    echo -e "|  print help information:                                                                                                                             |"
    echo -e "|     ./lite/tools/build_linux.sh help                                                                                                                 |"
    echo -e "|                                                                                                                                                      |"
    echo -e "|  optional argument:                                                                                                                                  |"
    echo -e "|     --arch: (armv8|armv7hf|armv7|x86), default is armv8                                                                                              |"
    echo -e "|     --toolchain: (gcc|clang), defalut is gcc                                                                                                         |"
    echo -e "|     --with_extra: (OFF|ON); controls whether to publish extra operators and kernels for (sequence-related model such as OCR or NLP), default is OFF  |"
    echo -e "|     --with_python: (OFF|ON); controls whether to build python lib or whl, default is OFF                                                             |"
    echo -e "|     --python_version: (2.7|3.5|3.7); controls python version to compile whl, default is None                                                         |"
    echo -e "|     --with_static_lib: (OFF|ON); controls whether to publish c++ api static lib, default is OFF                                                      |"
    echo -e "|     --with_cv: (OFF|ON); controls whether to compile cv functions into lib, default is OFF                                                           |"
    echo -e "|     --with_log: (OFF|ON); controls whether to print log information, default is ON                                                                   |"
    echo -e "|     --with_exception: (OFF|ON); controls whether to throw the exception when error occurs, default is OFF                                            |"
    echo -e "|     --with_profile: (OFF|ON); controls whether to profile speed, default is OFF                                                                      |"
    echo -e "|     --with_precision_profile: (OFF|ON); controls whether to profile precision, default is OFF                                                        |"
    echo -e "|     --with_benchmark: (OFF|ON); controls whether to compile benchmark binary, default is OFF                                                         |"
    echo -e "|                                                                                                                                                      |"
    echo -e "|  arguments of benchmark binary compiling:                                                                                                            |"
    echo -e "|     ./lite/tools/build_qnx.sh --with_benchmark=ON full_publish                                                                                     |"
    echo -e "|                                                                                                                                                      |"
    echo -e "|  arguments of striping lib according to input model:                                                                                                 |"
    echo -e "|     ./lite/tools/build_qnx.sh --with_strip=ON --opt_model_dir=YourOptimizedModelDir                                                                |"
    echo -e "|     --with_strip: (OFF|ON); controls whether to strip lib accrding to input model, default is OFF                                                    |"
    echo -e "|     --opt_model_dir: (absolute path to optimized model dir) required when compiling striped library                                                  |"
    echo -e "|  detailed information about striping lib:  https://paddle-lite.readthedocs.io/zh/latest/user_guides/library_tailoring.html                           |"
    echo -e "|                                                                                                                                                      |"
    echo -e "|  arguments of opencl library compiling:                                                                                                              |"
    echo -e "|     ./lite/tools/build_linux.sh --with_opencl=ON                                                                                                     |"
    echo -e "|     --with_opencl: (OFF|ON); controls whether to compile lib for opencl, default is OFF                                                              |"
    echo -e "|                                                                                                                                                      |"
    echo "--------------------------------------------------------------------------------------------------------------------------------------------------------"
    echo
}

function main {
    # Parse command line.
    for i in "$@"; do
        case $i in
            # armv8 or armv7hf or armv7, default armv8
            --arch=*)
                ARCH="${i#*=}"
                shift
                ;;
            # gcc or clang, default gcc
            --toolchain=*)
                TOOLCHAIN="${i#*=}"
                shift
                ;;
            # ON or OFF, default OFF
            --with_extra=*)
                WITH_EXTRA="${i#*=}"
                shift
                ;;
            # controls whether to compile cplus static library, default is OFF
            --with_static_lib=*)
                WITH_STATIC_LIB="${i#*=}"
                shift
                ;;
            # ON or OFF, default OFF
            --with_python=*)
                WITH_PYTHON="${i#*=}"
                shift
                ;;
            # 2.7 or 3.5 or 3.7, default is None
            --python_version=*)
                PY_VERSION="${i#*=}"
                shift
                ;;
            # ON or OFF, default OFF
            --with_cv=*)
                WITH_CV="${i#*=}"
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
                shift
                ;;
            # ON or OFF, default OFF
            --with_strip=*)
                BUILD_TAILOR="${i#*=}"
                shift
                ;;
            # string, absolute path to optimized model dir
            --opt_model_dir=*)
                OPTMODEL_DIR="${i#*=}"
                shift
                ;;
            # compiling lib which can operate on opencl and cpu.
            --with_opencl=*)
                WITH_OPENCL="${i#*=}"
                shift
                ;;
            # compiling lib which can operate on nnadapter.
            --with_nnadapter=*)
                WITH_NNADAPTER="${i#*=}"
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
            --nnadapter_with_fake_device=*)
                NNADAPTER_WITH_FAKE_DEVICE="${i#*=}"
                shift
                ;;
            --nnadapter_fake_device_sdk_root=*)
                NNADAPTER_FAKE_DEVICE_SDK_ROOT="${i#*=}"
                shift
                ;;
            # controls whether to include FP16 kernels, default is OFF
            --with_arm82_fp16=*)
                BUILD_ARM82_FP16="${i#*=}"
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
            # compiling lib with benchmark feature, default OFF.
            --with_benchmark=*)
                WITH_BENCHMARK="${i#*=}"
                shift
                ;;
            # ON or OFF, default OFF
            --with_train=*)
                WITH_TRAIN="${i#*=}"
                shift
                ;;
            # compiling result contains both light_api and cxx_api lib.
            full_publish)
                WITH_TINY_PUBLISH=OFF
                make_publish_so
                exit 0
                ;;
            # print help info
            help)
                print_usage
                exit 0
                ;;
            # unknown option
            *)
                echo "Error: unsupported argument \"${i#*=}\""
                print_usage
                exit 1
                ;;
        esac
    done

    make_publish_so
}

main $@
