#!/bin/bash
set -ex

# global variables with default value
if [   $NEUWARE_HOME   ];
then
  NEUWARE_HOME="${NEUWARE_HOME}"
else
  NEUWARE_HOME="/usr/local/neuware/"
fi
TARGET_NAME="all"    # default target
BUILD_EXTRA=ON                      # ON(with sequence ops)/OFF
WITH_TESTING=ON                     # ON/OFF

function print_usage {
    echo -e "\nUSAGE:"
    echo
    echo "----------------------------------------"
    echo -e "--mlu_sdk_root=<mlu sdk directory>"
    echo -e "--target_name=<target name>"
    echo "----------------------------------------"
    echo
}

# readonly variables with default value
readonly CMAKE_COMMON_OPTIONS="-DWITH_LITE=ON \
                               -DLITE_WITH_LIGHT_WEIGHT_FRAMEWORK=OFF \
                               -DLITE_WITH_ARM=OFF"

readonly NUM_CORES_FOR_COMPILE=${LITE_BUILD_THREADS:-8}

readonly THIRDPARTY_TAR=https://paddle-inference-dist.bj.bcebos.com/PaddleLite/third-party-05b862.tar.gz
readonly workspace=$(pwd)

function prepare_thirdparty {
    if [ ! -d $workspace/third-party ]; then
        rm -rf $workspace/third-party
    fi
    if [ ! -f $workspace/third-party-05b862.tar.gz ]; then
        wget $THIRDPARTY_TAR
    fi
    tar xvf third-party-05b862.tar.gz
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
    # cp ../${DEBUG_TOOL_PATH_PREFIX}/analysis_tool.py ./${DEBUG_TOOL_PATH_PREFIX}/

    # clone submodule
    # git submodule update --init --recursive
    prepare_thirdparty
}

function build_mlu {
    prepare_workspace
    build_dir=${workspace}/build.lite.mlu
    mkdir -p $build_dir
    cd $build_dir

    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$PWD/third_party/install/mklml/lib"
    cmake .. \
        ${CMAKE_COMMON_OPTIONS} \
        -DWITH_GPU=OFF \
        -DWITH_MKLDNN=OFF \
        -DLITE_WITH_X86=ON \
        -DWITH_MKL=ON \
        -DLITE_WITH_MLU=ON \
        -DLITE_WITH_PYTHON=OFF \
        -DLITE_BUILD_EXTRA=${BUILD_EXTRA} \
        -DWITH_TESTING=${WITH_TESTING} \
        -DNEUWARE_HOME=${NEUWARE_HOME}

    make $TARGET_NAME -j$NUM_CORES_FOR_COMPILE

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
            --build_extra=*)
                BUILD_EXTRA="${i#*=}"
                shift
                ;;
            --neuware_home=*)
                NEUWARE_HOME="${i#*=}"
                shift
                ;;
            build)
                build_mlu
                shift
                ;;
            full_publish)
                TARGET_NAME=publish_inference
                build_mlu
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
