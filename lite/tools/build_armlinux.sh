#!/bin/bash
set -ex

#####################################################################################################
# 1. global variables, you can change them according to your requirements
#####################################################################################################
# armv7 or armv8 or armv7hf, default armv8.
ARM_ABI=armv8
# gcc or clang, default gcc.
ARM_LANG=gcc
# ON or OFF, default OFF.
BUILD_EXTRA=OFF
# controls whether to compile python lib, default is OFF.
BUILD_PYTHON=OFF
# controls whether to compile cv functions into lib, default is OFF.
BUILD_CV=OFF
# controls whether to hide log information, default is ON.
SHUTDOWN_LOG=ON
# options of striping lib according to input model.
BUILD_TAILOR=OFF
OPTMODEL_DIR=""
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
function init_cmake_common_options {
    CMAKE_COMMON_OPTIONS="-DWITH_LITE=ON \
                        -DLITE_WITH_ARM=ON \
                        -DLITE_WITH_X86=OFF \
                        -DARM_TARGET_OS=armlinux \
                        -DLITE_WITH_LIGHT_WEIGHT_FRAMEWORK=ON \
                        -DLITE_BUILD_EXTRA=$BUILD_EXTRA \
                        -DLITE_WITH_CV=$BUILD_CV \
                        -DLITE_WITH_PYTHON=$BUILD_PYTHON \
                        -DLITE_SHUTDOWN_LOG=$SHUTDOWN_LOG \
                        -DLITE_BUILD_TAILOR=$BUILD_TAILOR \
                        -DLITE_OPTMODEL_DIR=$OPTMODEL_DIR \
                        -DARM_TARGET_ARCH_ABI=$ARM_ABI \
                        -DARM_TARGET_LANG=$ARM_LANG"
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
    build_dir=$workspace/build.lite.armlinux.$ARM_ABI.$ARM_LANG
    if [ -d $build_dir ]
    then
        rm -rf $build_dir
    fi
    mkdir -p $build_dir
    cd $build_dir

    init_cmake_common_options
    cmake $workspace \
        ${CMAKE_COMMON_OPTIONS} \
        -DLITE_ON_TINY_PUBLISH=ON

    make publish_inference -j$NUM_PROC
    cd - > /dev/null
}
####################################################################################################

# 4.2 function of full_publish compiling
# here we compile both light_api lib and full_api lib
function make_full_publish_so {
    prepare_thirdparty

    build_directory=$BUILD_DIR/build.lite.android.$ARM_ABI.$ARM_LANG
    if [ -d $build_directory ]
    then
        rm -rf $build_directory
    fi
    mkdir -p $build_directory
    cd $build_directory

    prepare_workspace $workspace $build_directory

    init_cmake_common_options
    cmake $workspace \
        ${CMAKE_COMMON_OPTIONS} \
        -DLITE_ON_TINY_PUBLISH=OFF

    make publish_inference -j$NUM_PROC
    cd - > /dev/null
}
####################################################################################################

# 4.3 function of opencl compiling
# here we compile both light_api and full_api opencl lib
function make_opencl {
    prepare_thirdparty

    build_dir=$workspace/build.lite.android.$ARM_ABI.$ARM_LANG.opencl
    if [ -d $build_directory ]
    then
    rm -rf $build_directory
    fi
    mkdir -p $build_dir
    cd $build_dir

    prepare_workspace $workspace $build_dir
    prepare_opencl_source_code $workspace $build_dir

    cmake $workspace \
        ${CMAKE_COMMON_OPTIONS} \
        -DLITE_WITH_OPENCL=ON

    make opencl_clhpp -j$NUM_PROC
    make publish_inference -j$NUM_PROC
}
####################################################################################################

function print_usage {
    set +x
    echo -e "\n Methods of compiling Padddle-Lite armlinux library:"
    echo "----------------------------------------"
    echo -e "compile light_api library (recommanded): (armv8, gcc)"
    echo -e "   ./lite/tools/build_armlinux.sh tiny_publish"
    echo -e "compile both light_api and cxx_api library: (armv8, gcc)"
    echo -e "   ./lite/tools/build_armlinux.sh full_publish"
    echo -e "compile both light_api and cxx_api opencl library: (armv8, gcc)"
    echo -e "   ./lite/tools/build_armlinux.sh opencl"
    echo
    echo -e "optional argument:"
    echo -e "--arm_abi:\t armv8|armv7|armv7hf, default is armv8"
    echo -e "--arm_lang:\t gcc|clang, defalut is gcc"
    echo -e "--build_python: (OFF|ON); controls whether to publish python api lib, default is OFF"
    echo -e "--build_cv: (OFF|ON); controls whether to compile cv functions into lib, default is OFF"
    echo -e "--shutdown_log: (OFF|ON); controls whether to hide log information, default is ON"
    echo -e "--build_extra: (OFF|ON); controls whether to publish extra operators and kernels for (sequence-related model such as OCR or NLP)"
    echo
    echo -e "arguments of striping lib according to input model:"
    echo -e "--build_tailor: (OFF|ON); controls whether to tailor the lib according to the model, default is OFF"
    echo -e "--opt_model_dir: (path to optimized model dir); contains absolute path to optimized model dir"
    echo "----------------------------------------"
    echo
}

function main {
    if [ -z "$1" ]; then
        # compiling result contains light_api lib only, recommanded.
        make_tiny_publish_so
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
                make_tiny_publish_so
                shift
                ;;
            # compiling result contains both light_api and cxx_api lib.
            full_publish)
                make_full_publish_so
                shift
                ;;
            # compiling lib which can operate on opencl and cpu.
            opencl)
                make_opencl
                shift
                ;;
            *)
                # unknown option
                print_usage
                exit 1
                ;;
        esac
        # compiling result contains light_api lib only, recommanded.
        make_tiny_publish_so
    done
}

main $@
