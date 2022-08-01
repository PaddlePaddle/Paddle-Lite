#!/bin/bash
set -ex

# global variables with default value
BM_SDK_ROOT="$(pwd)/third-party/bmlibs/bm_sc3_libs"     # BM SDK
TARGET_NAME="BM1682"     # default target
BUILD_EXTRA=OFF                  # ON(with sequence ops)/OFF
WITH_TESTING=ON                  # ON/OFF
BM_DYNAMIC_COMPILE=OFF
BM_SAVE_UMODEL=OFF
BM_SAVE_BMODEL=OFF

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

# url that stores third-party tar.gz file to accelerate third-party lib installation
readonly THIRDPARTY_URL=https://paddlelite-data.bj.bcebos.com/third_party_libs/
readonly THIRDPARTY_TAR=third-party-801f670.tar.gz
readonly workspace=$(pwd)

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
}

function build_bm {
    build_dir=${workspace}/build.lite.bm
    prepare_thirdparty
    mkdir -p $build_dir
    cd $build_dir

    if [ $TARGET_NAME == "BM1684" ]; then
      BM_SDK_ROOT="$workspace/third-party/bmlibs/bm_sc5_libs"
    else
      BM_SDK_ROOT="$workspace/third-party/bmlibs/bm_sc3_libs"
    fi
    echo $BM_SDK_ROOT

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
        -DLITE_ON_TINY_PUBLISH=OFF \
        -DWITH_TESTING=${WITH_TESTING} \
        -DBM_DYNAMIC_COMPILE=${BM_DYNAMIC_COMPILE} \
        -DBM_SAVE_UMODEL=${BM_SAVE_UMODEL} \
        -DBM_SAVE_BMODEL=${BM_SAVE_BMODEL} \
        -DBM_SDK_ROOT=${BM_SDK_ROOT}

    make publish_inference -j$NUM_CORES_FOR_COMPILE

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
            --test=*)
                WITH_TESTING=${i#*=}
                shift
                ;;
            --dynamic=*)
                BM_DYNAMIC_COMPILE=${i#*=}
                shift
                ;;
            --save_bmodel=*)
                BM_SAVE_BMODEL=${i#*=}
                shift
                ;;
            --save_umodel=*)
                BM_SAVE_UMODEL=${i#*=}
                shift
                ;;
            *)
                # unknown option
                print_usage
                exit 1
                ;;
        esac
    done
    build_bm
}
main $@
