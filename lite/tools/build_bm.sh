#!/bin/bash
set -ex

# global variables with default value
BM_SDK_ROOT="$(pwd)/third-party/bmlibs/bm_sc3_libs"     # BM SDK
TARGET_NAME="BM1682"     # default target
BUILD_EXTRA=OFF                     # ON(with sequence ops)/OFF
WITH_TESTING=ON                    # ON/OFF

function print_usage {
    echo -e "\nUSAGE:"
    echo
    echo "----------------------------------------"
    echo -e "--bm_sdk_root=<bm sdk directory>"
    echo -e "--target_name=<target name>"
    echo "----------------------------------------"
    echo
}

# readonly variables with default value
readonly CMAKE_COMMON_OPTIONS="-DWITH_LITE=ON \
                               -DLITE_WITH_LIGHT_WEIGHT_FRAMEWORK=OFF \
                               -DWITH_PYTHON=OFF \
                               -DLITE_WITH_ARM=OFF"

readonly NUM_CORES_FOR_COMPILE=${LITE_BUILD_THREADS:-1}

readonly THIRDPARTY_TAR=https://paddle-inference-dist.bj.bcebos.com/PaddleLite/third-party-05b862.tar.gz
readonly workspace=$(pwd)

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

    # clone bmlibs
    if [ ! -d ${workspace}/third-party/bmlibs ]; then
        git clone https://github.com/AnBaolei1984/bmlibs.git ${workspace}/third-party/bmlibs
    fi     
}

# for code gen, a source file is generated after a test, but is dependended by some targets in cmake.
# here we fake an empty file to make cmake works.
function prepare_workspace {
    # in build directory
    # 1. Prepare gen_code file
    GEN_CODE_PATH_PREFIX=lite/gen_code
    mkdir -p ./${GEN_CODE_PATH_PREFIX}
    touch ./${GEN_CODE_PATH_PREFIX}/__generated_code__.cc

    # 2.Prepare debug tool
    DEBUG_TOOL_PATH_PREFIX=lite/tools/debug
    mkdir -p ./${DEBUG_TOOL_PATH_PREFIX}
    cp ../${DEBUG_TOOL_PATH_PREFIX}/analysis_tool.py ./${DEBUG_TOOL_PATH_PREFIX}/

    # clone submodule
    # git submodule update --init --recursive
    prepare_thirdparty
}

function build_bm {
    build_dir=${workspace}/build.lite.bm
    mkdir -p $build_dir
    cd $build_dir

    prepare_workspace
    cmake .. \
        ${CMAKE_COMMON_OPTIONS} \
        -DWITH_GPU=OFF \
        -DWITH_MKLDNN=OFF \
        -DLITE_WITH_X86=OFF \
        -DWITH_MKL=OFF \
        -DLITE_BUILD_EXTRA=ON \
        -DLITE_WITH_XPU=OFF \
        -DLITE_WITH_BM=ON \
        -DWITH_TESTING=${WITH_TESTING} \
        -DBM_SDK_ROOT=${BM_SDK_ROOT}

    make -j$NUM_CORES_FOR_COMPILE

    cd -
    echo "Done"
}

function main {
    # Parse command line.
    for i in "$@"; do
        case $i in
            --target_name=*)
                TARGET_NAME="${i#*=}"
                shift
                ;;
            #--bm_sdk_root=*)
            #    BM_SDK_ROOT="${i#*=}"
            #    shift
            #    ;;
            bm)
                build_bm
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
