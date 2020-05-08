#!/bin/bash
set -ex

# global variables with default value
ARM_OS="android"                    # android only yet
ARM_ABI="armv8"                     # armv8, armv7
ARM_LANG="gcc"                      # gcc only yet
ANDROID_STL="c++_shared"            # c++_shared/c++_static, c++_shared is used by HiAI DDK 310
DDK_ROOT="$(pwd)/ai_ddk_lib/"       # HiAI DDK 310 from https://developer.huawei.com/consumer/cn/hiai/
TARGET_NAME="test_subgraph_pass"    # default target
BUILD_EXTRA=OFF                     # ON(with sequence ops)/OFF
WITH_JAVA=ON                        # ON(build jar and jni so)/OFF
WITH_TESTING=ON                     # ON/OFF
WITH_LOG=ON                         # ON(disable logging)/OFF
ON_TINY_PUBLISH=OFF                 # ON(tiny publish)/OFF(full publish)

function print_usage {
    echo -e "\nUSAGE:"
    echo
    echo "----------------------------------------"
    echo -e "--arm_os=<os> android only yet."
    echo -e "--arm_abi=<abi> armv8, armv7 yet."
    echo -e "--android_stl=<shared> c++_shared or c++_static"
    echo -e "--arm_lang=<gcc>"
    echo -e "--ddk_root=<hiai_ddk_root>"
    echo -e "--target_name=<target_name>"
    echo "----------------------------------------"
    echo
}

# for code gen, a source file is generated after a test, 
# but is dependended by some targets in cmake.
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
}

function prepare_thirdparty {
    readonly THIRDPARTY_TAR=https://paddle-inference-dist.bj.bcebos.com/PaddleLite/third-party-05b862.tar.gz

    readonly workspace=$PWD
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

function build_npu {
    cur_dir=$(pwd)

    prepare_thirdparty

    local stl_dir
    local publish_dir
    # the c++ symbol is not recognized by the bundled script
    if [[ "${ANDROID_STL}" == "c++_shared" ]]; then
        stl_dir="cxx_shared"
    fi
    if [[ "${ANDROID_STL}" == "c++_static" ]]; then
        stl_dir="cxx_static"
    fi
    if [[ "${ON_TINY_PUBLISH}" == "ON" ]]; then
        WITH_TESTING=OFF
        WITH_LOG=OFF
        publish_dir="tiny_publish"
    else
        publish_dir="full_publish"
    fi
    build_dir=$cur_dir/build.lite.npu.${ARM_OS}.${ARM_ABI}.${ARM_LANG}.${stl_dir}.${publish_dir}
    mkdir -p $build_dir
    cd $build_dir

    # NPU libs need API LEVEL 24 above
    prepare_workspace
    cmake .. \
        -DWITH_GPU=OFF \
        -DWITH_MKL=OFF \
        -DWITH_LITE=ON \
        -DLITE_WITH_CUDA=OFF \
        -DLITE_WITH_X86=OFF \
        -DLITE_BUILD_EXTRA=${BUILD_EXTRA} \
        -DLITE_WITH_ARM=ON \
        -DWITH_ARM_DOTPROD=ON   \
        -DLITE_WITH_LIGHT_WEIGHT_FRAMEWORK=ON \
        -DWITH_TESTING=${WITH_TESTING} \
        -DLITE_WITH_JAVA=${WITH_JAVA} \
        -DLITE_WITH_LOG=${WITH_LOG} \
        -DLITE_WITH_NPU=ON \
        -DLITE_ON_TINY_PUBLISH=${ON_TINY_PUBLISH} \
        -DANDROID_API_LEVEL=24 \
        -DARM_TARGET_OS=${ARM_OS} \
        -DARM_TARGET_ARCH_ABI=${ARM_ABI} \
        -DARM_TARGET_LANG=${ARM_LANG} \
        -DANDROID_STL_TYPE=${ANDROID_STL} \
        -DNPU_DDK_ROOT=${DDK_ROOT}

    make $TARGET_NAME -j2

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
            --ddk_root=*)
                DDK_ROOT="${i#*=}"
                shift
                ;;
            build)
                build_npu
                shift
                ;;
            full_publish)
                TARGET_NAME=publish_inference
                build_npu
                shift
                ;;
            tiny_publish)
                ON_TINY_PUBLISH=ON
                TARGET_NAME=publish_inference
                build_npu
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
