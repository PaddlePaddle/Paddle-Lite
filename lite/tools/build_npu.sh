#!/bin/bash
set -ex

function print_usage {
    echo -e "\nUSAGE:"
    echo
    echo "----------------------------------------"
    echo -e "--arm_os=<os> android only yet."
    echo -e "--arm_abi=<abi> armv8, armv7 yet."
    echo -e "--android_stl=<shared> shared or static"
    echo -e "--arm_lang=<gcc> "
    echo -e "--ddk_root=<hiai_ddk_root> "
    echo -e "--test_name=<test_name>"
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

function cmake_npu {
    prepare_workspace
    # $1: ARM_TARGET_OS in "android" , "armlinux"
    # $2: ARM_TARGET_ARCH_ABI in "armv8", "armv7" ,"armv7hf"
    # $3: ARM_TARGET_LANG in "gcc" "clang"
    # $4: ANDROID_STL_TYPE in "c++_shared" "c++_static"
    # $5: DDK_ROOT path

    # NPU libs need API LEVEL 24 above
    cmake .. \
        -DWITH_GPU=OFF \
        -DWITH_MKL=OFF \
        -DWITH_LITE=ON \
        -DLITE_WITH_CUDA=OFF \
        -DLITE_WITH_X86=OFF \
        -DLITE_BUILD_EXTRA=ON \
        -DLITE_WITH_ARM=ON \
        -DWITH_ARM_DOTPROD=ON   \
        -DLITE_WITH_LIGHT_WEIGHT_FRAMEWORK=ON \
        -DWITH_TESTING=ON \
        -DLITE_WITH_JAVA=ON \
        -DLITE_WITH_NPU=ON \
        -DANDROID_API_LEVEL=24 \
        -DARM_TARGET_OS=$1 \
        -DARM_TARGET_ARCH_ABI=$2 \
        -DARM_TARGET_LANG=$3 \
        -DANDROID_STL_TYPE=$4 \
        -DNPU_DDK_ROOT=$5
}

function build_npu {
    # os, abi, lang, stl, ddk_root, test_name
    cur_dir=$(pwd)

    local os=android
    local abi=armv8
    local lang=gcc
    local stl="c++_shared"
    local ddk_root="${cur_dir}/ai_ddk_lib/" 
    local test_name=test_npu_pass
    prepare_thirdparty

    if [ "x${ARM_OS}" != "x" ]; then
        os=$ARM_OS
    fi
    if [[ "x${ARM_ABI}" != "x" ]]; then
        abi=$ARM_ABI
    fi
    if [[ "x${ARM_LANG}" != "x" ]]; then
        lang=$ARM_LANG
    fi
    if [[ "x${ANDROID_STL}" != "x" ]]; then
        stl=$ANDROID_STL
    fi
    if [[ "x${DDK_ROOT}" != "x" ]]; then
        ddk_root=$DDK_ROOT
    fi
    if [[ $# -ge 1 ]]; then
        test_name=$1
    fi

    # the c++ symbol is not recognized by the bundled script
    if [[ "${stl}" == "c++_shared" ]]; then
        stl_dir="cxx_shared"
    fi
    if [[ "${stl}" == "c++_static" ]]; then
        stl_dir="cxx_static"
    fi
    build_dir=$cur_dir/build.lite.npu.${os}.${abi}.${lang}.${stl_dir}
    mkdir -p $build_dir
    cd $build_dir

    cmake_npu ${os} ${abi} ${lang} ${stl} ${ddk_root}
    make $test_name -j8

    cd -
    echo "Done"
}

function main {
    # Parse command line.
    for i in "$@"; do
        case $i in
            --tests=*)
                TESTS_FILE="${i#*=}"
                shift
                ;;
            --test_name=*)
                TEST_NAME="${i#*=}"
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
            --ddk_root=*)
                DDK_ROOT="${i#*=}"
                shift
                ;;
            build)
                build_npu $TEST_NAME
                shift
                ;;
            full_publish)
                build_npu publish_inference
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
