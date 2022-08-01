#!/bin/bash
set -ex

# global variables with default value
ARM_OS="armlinux"                    # android only yet
ARM_ABI="armv8"                     # armv8, armv7
ARM_LANG="gcc"                      # gcc only yet
DDK_ROOT="$(pwd)/rknpu"
TARGET_NAME="test_subgraph_pass"    # default target
BUILD_EXTRA=OFF                     # ON(with sequence ops)/OFF
WITH_TESTING=ON     	            # ON/OFF
WITH_LOG=ON                         # ON(disable logging)/OFF
ON_TINY_PUBLISH=OFF                 # ON(tiny publish)/OFF(full publish)

function print_usage {
    echo -e "\nUSAGE:"
    echo
    echo "----------------------------------------"
    echo -e "--arm_os=<os> android only yet."
    echo -e "--arm_abi=<abi> armv8, armv7 yet."
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
    # url that stores third-party tar.gz file to accelerate third-party lib installation
    readonly THIRDPARTY_URL=https://paddlelite-data.bj.bcebos.com/third_party_libs/
    readonly THIRDPARTY_TAR=third-party-801f670.tar.gz

    readonly workspace=$PWD
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

function build_npu {
    cur_dir=$(pwd)

    prepare_thirdparty

    local publish_dir
    if [[ "${ON_TINY_PUBLISH}" == "ON" ]]; then
        WITH_TESTING=OFF
        WITH_LOG=OFF
        publish_dir="tiny_publish"
    else
        publish_dir="full_publish"
    fi
    build_dir=$cur_dir/build.lite.rknpu.${ARM_OS}.${ARM_ABI}.${ARM_LANG}.${publish_dir}
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
        -DLITE_WITH_NPU=OFF \
        -DLITE_WITH_JAVA=OFF \
        -DLITE_WITH_ARM=ON \
        -DLITE_WITH_LIGHT_WEIGHT_FRAMEWORK=ON	\
        -DWITH_ARM_DOTPROD=ON   \
        -DLITE_BUILD_EXTRA=${BUILD_EXTRA} \
        -DWITH_TESTING=${WITH_TESTING} \
        -DLITE_WITH_LOG=${WITH_LOG} \
        -DLITE_ON_TINY_PUBLISH=${ON_TINY_PUBLISH} \
        -DARM_TARGET_OS=${ARM_OS} \
        -DARM_TARGET_ARCH_ABI=${ARM_ABI} \
        -DARM_TARGET_LANG=${ARM_LANG} \
        -DLITE_WITH_RKNPU=ON \
        -DRKNPU_DDK_ROOT=${DDK_ROOT}

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
