#!/bin/bash
# The git version of CI is 2.7.4. This script is not compatible with git version 1.7.1.
set -ex

TESTS_FILE="./lite_tests.txt"
LIBS_FILE="./lite_libs.txt"
LITE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../" && pwd)"

# url that stores third-party tar.gz file to accelerate third-party lib installation
readonly THIRDPARTY_URL=https://paddlelite-data.bj.bcebos.com/third_party_libs/
readonly THIRDPARTY_TAR=third-party-91a9ab3.tar.gz
readonly workspace=$PWD

NUM_CORES_FOR_COMPILE=${LITE_BUILD_THREADS:-8}
ROOT_DIR=$(pwd)
BUILD_DIRECTORY=$(pwd)/build

# Global variables
# The list of os for building unit tests(android,armlinux), such as "android"
OS_LIST="android"
# The list of arch abi for building unit tests(armv8,armv7,armv7hf), such as "armv8,armv7"
# for android devices, "armv8" for RK3399, "armv7hf" for Raspberry pi 3B
ARCH_LIST="armv8,armv7"
# The list of toolchains for building unit tests(gcc,clang), such as "gcc,clang"
# for android, "gcc" for armlinx
TOOLCHAIN_LIST="gcc,clang"
# The list of unit tests to be checked, use commas to separate them, such as "test_cxx_api,test_mobilenetv1_int8"
UNIT_TEST_CHECK_LIST="test_light_api,test_apis,test_paddle_api,test_cxx_api,test_vector_view"
# 0: black list 1: white list
UNIT_TEST_FILTER_TYPE=0
# The Logging level of GLOG for unit tests
UNIT_TEST_LOG_LEVEL=5
# Remote device type(0: adb, 1: ssh) for real android and armlinux devices
REMOTE_DEVICE_TYPE=0
# The list of the device names for the real android devices, use commas to separate them, such as "2GX0119401000796,0123456789ABCDEF,5aba5d0ace0b89f6"
# The list of the device infos for the real armlinux devices, its format is "dev0_ip_addr,dev0_usr_id,dev0_usr_pwd:dev1_ip_addr,dev1_usr_id,dev1_usr_pwd"
REMOTE_DEVICE_LIST="2GX0119401000796,0123456789ABCDEF"
# Work directory of the remote devices for running the unit tests
REMOTE_DEVICE_WORK_DIR="/data/local/tmp"
# Xpu sdk option
XPU_SDK_URL=""
XPU_XDNN_URL=""
XPU_XRE_URL=""
XPU_SDK_ENV=""
XPU_SDK_ROOT=""

# if operating in mac env, we should expand the maximum file num
os_name=$(uname -s)
if [ ${os_name} == "Darwin" ]; then
    ulimit -n 1024
fi

function prepare_thirdparty {
    cd $workspace
    if [ ! -d $workspace/third-party -o -f $workspace/$THIRDPARTY_TAR ]; then
        rm -rf $workspace/third-party

        if [ ! -f $workspace/$THIRDPARTY_TAR ]; then
            wget $THIRDPARTY_URL/$THIRDPARTY_TAR
        fi
        tar xzf $THIRDPARTY_TAR
    else
        git submodule update --init --recursive
    fi
    cd -
}

# for code gen, a source file is generated after a test, but is dependended by some targets in cmake.
# here we fake an empty file to make cmake works.
function prepare_workspace() {
    local root_dir_=$1
    local build_dir_=$2

    # in build directory
    # 1. Prepare gen_code file
    GEN_CODE_PATH_PREFIX=$build_dir_/lite/gen_code
    mkdir -p ${GEN_CODE_PATH_PREFIX}
    touch ${GEN_CODE_PATH_PREFIX}/__generated_code__.cc

    # 2.Prepare debug tool
    DEBUG_TOOL_PATH_PREFIX=$build_dir_/lite/tools/debug
    mkdir -p ${DEBUG_TOOL_PATH_PREFIX}
    cp $root_dir_/lite/tools/debug/analysis_tool.py ${DEBUG_TOOL_PATH_PREFIX}/

    prepare_thirdparty
}

function adb_device_check() {
    local adb_device_name=$1
    if [[ -n "$adb_device_name" ]]; then
        for line in $(adb devices | grep -v "List" | awk '{print $1}'); do
            online_device_name=$(echo $line | awk '{print $1}')
            if [[ "$adb_device_name" == "$online_device_name" ]]; then
                return 0
            fi
        done
    fi
    return 1
}

function adb_device_pick() {
    local adb_device_list=$1
    local adb_device_names=(${adb_device_list//,/ })
    for adb_device_name in ${adb_device_names[@]}; do
        adb_device_check $adb_device_name
        if [[ $? -eq 0 ]]; then
            echo $adb_device_name
            return 0
        fi
    done
    echo ""
    return 1
}

function adb_device_run() {
    local adb_device_name=$1
    local adb_device_cmd=$2
    if [[ "$adb_device_cmd" == "shell" ]]; then
        adb -s $adb_device_name shell "$3"
    elif [[ "$adb_device_cmd" == "push" ]]; then
        local src_path=$3
        local dst_path=$4
        # adb push don't support '/*', so replace it with '/.'
        if [[ ${#src_path} -gt 2 ]]; then
            local src_suffix=${src_path: -2}
            if [[ "$src_suffix" == "/*" ]]; then
                src_path=${src_path:0:-2}/.
            fi
        fi
        adb -s $adb_device_name push "$src_path" "$dst_path"
    elif [[ "$adb_device_cmd" == "root" ]]; then
        adb -s $adb_device_name root
    elif [[ "$adb_device_cmd" == "remount" ]]; then
        adb -s $adb_device_name remount
    else
        echo "Unknown command $adb_device_cmd!"
    fi
}

function ssh_device_check() {
    local ssh_device_name=$1
    ssh_device_run $ssh_device_name test
}

function ssh_device_pick() {
    local ssh_device_list=$1
    local ssh_device_names=(${ssh_device_list//:/ })
    for ssh_device_name in ${ssh_device_names[@]}; do
        ssh_device_check $ssh_device_name
        if [[ $? -eq 0 ]]; then
            echo $ssh_device_name
            return 0
        fi
    done
    echo ""
    return 1
}

function ssh_device_run() {
    local ssh_device_name=$1
    local ssh_device_cmd=$2
    if [[ -z "$ssh_device_name" ]]; then
        echo "SSH device name is empty!"
        exit 1
    fi
    local ssh_device_items=(${ssh_device_name//,/ })
    if [[ ${#ssh_device_items[@]} -ne 3 ]]; then
        echo "SSH device name parse failed!"
        exit 1
    fi
    local ssh_device_ip_addr=${ssh_device_items[0]}
    local ssh_device_usr_id=${ssh_device_items[1]}
    local ssh_device_usr_pwd=${ssh_device_items[2]}
    if [[ -z "$ssh_device_ip_addr" || -z "$ssh_device_usr_id" ]]; then
        echo "SSH device IP Address or User ID is empty!"
        exit 1
    fi
    if [[ "$ssh_device_cmd" == "shell" ]]; then
        sshpass -p $ssh_device_usr_pwd ssh -o ConnectTimeout=60 -o StrictHostKeyChecking=no $ssh_device_usr_id@$ssh_device_ip_addr "$3"
    elif [[ "$ssh_device_cmd" == "push" ]]; then
        sshpass -p $ssh_device_usr_pwd scp -r -o ConnectTimeout=60 -o StrictHostKeyChecking=no $3 $ssh_device_usr_id@$ssh_device_ip_addr:$4
    elif [[ "$ssh_device_cmd" == "test" ]]; then
        sshpass -p $ssh_device_usr_pwd ssh -o ConnectTimeout=60 -o StrictHostKeyChecking=no $ssh_device_usr_id@$ssh_device_ip_addr "exit 0" &>/dev/null
    else
        echo "Unknown command $ssh_device_cmd!"
        exit 1
    fi
}

function run_unit_test_on_remote_device() {
    local remote_device_name=""
    local remote_device_work_dir=""
    local remote_device_check=""
    local remote_device_run=""
    local target_name=""
    local model_dir=""
    local data_dir=""
    local config_dir=""
    # Extract arguments from command line
    for i in "$@"; do
        case $i in
        --remote_device_name=*)
            remote_device_name="${i#*=}"
            shift
            ;;
        --remote_device_work_dir=*)
            remote_device_work_dir="${i#*=}"
            shift
            ;;
        --remote_device_check=*)
            remote_device_check="${i#*=}"
            shift
            ;;
        --remote_device_run=*)
            remote_device_run="${i#*=}"
            shift
            ;;
        --target_name=*)
            target_name="${i#*=}"
            shift
            ;;
        --model_dir=*)
            model_dir="${i#*=}"
            shift
            ;;
        --data_dir=*)
            data_dir="${i#*=}"
            shift
            ;;
        --config_dir=*)
            config_dir="${i#*=}"
            shift
            ;;
        *)
            shift
            ;;
        esac
    done

    # Be careful!!! Don't delete the root or system directories if the device is rooted.
    if [[ -z "$remote_device_work_dir" ]]; then
        echo "$remote_device_work_dir can't be empty!"
        exit 1
    fi
    if [[ "$remote_device_work_dir" == "/" ]]; then
        echo "$remote_device_work_dir can't be root dir!"
        exit 1
    fi

    # Copy the executable unit test to the remote device
    local target_path=$(find ./lite -name $target_name)
    if [[ -z "$target_path" ]]; then
        echo "$target_name not found!"
        exit 1
    fi
    $remote_device_run $remote_device_name shell "rm -f $remote_device_work_dir/$target_name"
    $remote_device_run $remote_device_name push "$target_path" "$remote_device_work_dir"

    local command_line="./$target_name"
    # Copy the model files to the remote device
    if [[ -n "$model_dir" ]]; then
        local model_name=$(basename $model_dir)
        $remote_device_run $remote_device_name shell "rm -rf $remote_device_work_dir/$model_name"
        $remote_device_run $remote_device_name push "$model_dir" "$remote_device_work_dir"
        command_line="$command_line --model_dir ./$model_name"
    fi

    # Copy the test data files to the remote device
    if [[ -n "$data_dir" ]]; then
        local data_name=$(basename $data_dir)
        $remote_device_run $remote_device_name shell "rm -rf $remote_device_work_dir/$data_name"
        $remote_device_run $remote_device_name push "$data_dir" "$remote_device_work_dir"
        command_line="$command_line --data_dir ./$data_name"
    fi

    # Copy the config files to the remote device
    if [[ -n "$config_dir" ]]; then
        local config_name=$(basename $config_dir)
        $remote_device_run $remote_device_name shell "rm -rf $remote_device_work_dir/$config_name"
        $remote_device_run $remote_device_name push "$config_dir" "$remote_device_work_dir"
        command_line="$command_line --config_dir ./$config_name"
    fi

    # Run the model on the remote device
    $remote_device_run $remote_device_name shell "ulimit -s unlimited; cd $remote_device_work_dir; export GLOG_v=$UNIT_TEST_LOG_LEVEL; LD_LIBRARY_PATH=$LD_LIBRARY_PATH:. $command_line"
}

function build_and_test_on_remote_device() {
    local os_list=$1
    local arch_list=$2
    local toolchain_list=$3
    local unit_test_check_list=$4
    local unit_test_filter_type=$5
    local build_target_func=$6
    local prepare_device_func=$7
    local remote_device_type=$8
    local remote_device_list=$9
    local remote_device_work_dir=${10}
    local extra_arguments=${11}

    # Set helper functions to access the remote devices
    local remote_device_pick=ssh_device_pick
    local remote_device_check=ssh_device_check
    local remote_device_run=ssh_device_run
    if [[ $remote_device_type -eq 0 ]]; then
        remote_device_pick=adb_device_pick
        remote_device_check=adb_device_check
        remote_device_run=adb_device_run
    fi

    # Pick the first available remote device from list
    local remote_device_name=$($remote_device_pick $remote_device_list)
    if [[ -z $remote_device_name ]]; then
        echo "No remote device available!"
        exit 1
    else
        echo "Found a device $remote_device_name."
    fi

    # Run all of unittests and model tests
    local oss=(${os_list//,/ })
    local archs=(${arch_list//,/ })
    local toolchains=(${toolchain_list//,/ })
    local unit_test_check_items=(${unit_test_check_list//,/ })
    for os in $oss; do
        for arch in $archs; do
            for toolchain in $toolchains; do
                # Build all tests and prepare device environment for running tests
                echo "Build tests with $arch+$toolchain"
                $build_target_func $os $arch $toolchain $extra_arguments
                $prepare_device_func $os $arch $toolchain $remote_device_name $remote_device_work_dir $remote_device_check $remote_device_run $extra_arguments
                # Run all of unit tests and model tests
                for test_name in $(cat $TESTS_FILE); do
                    local is_matched=0
                    for unit_test_check_item in ${unit_test_check_items[@]}; do
                        if [[ "$unit_test_check_item" == "$test_name" ]]; then
                            echo "$test_name on the checklist."
                            is_matched=1
                            break
                        fi
                    done
                    # black list
                    if [[ $is_matched -eq 1 && $unit_test_filter_type -eq 0 ]]; then
                        continue
                    fi
                    # white list
                    if [[ $is_matched -eq 0 && $unit_test_filter_type -eq 1 ]]; then
                        continue
                    fi
                    # Extract the arguments from ctest command line
                    test_cmds=$(ctest -V -N -R ^$test_name$)
                    reg_expr=".*Test command:.*\/$test_name \(.*\) Test #[0-9]*: $test_name.*"
                    test_args=$(echo $test_cmds | sed -n "/$reg_expr/p")
                    if [[ -n "$test_args" ]]; then
                        # Matched, extract and remove the quotes
                        test_args=$(echo $test_cmds | sed "s/$reg_expr/\1/g")
                        test_args=$(echo $test_args | sed "s/\"//g")
                    fi
                    run_unit_test_on_remote_device --remote_device_name=$remote_device_name --remote_device_work_dir=$remote_device_work_dir --remote_device_check=$remote_device_check --remote_device_run=$remote_device_run --target_name=$test_name $test_args
                done
                cd - >/dev/null
            done
        done
    done
}

# Android
function android_cpu_prepare_device() {
    local os=$1
    local arch=$2
    local toolchain=$3
    local remote_device_name=$4
    local remote_device_work_dir=$5
    local remote_device_check=$6
    local remote_device_run=$7

    # Check device is available
    $remote_device_check $remote_device_name
    if [[ $? -ne 0 ]]; then
        echo "$remote_device_name not found!"
        exit 1
    fi

    # Create work dir on the remote device
    if [[ -z "$remote_device_work_dir" ]]; then
        echo "$remote_device_work_dir can't be empty!"
        exit 1
    fi
    if [[ "$remote_device_work_dir" == "/" ]]; then
        echo "$remote_device_work_dir can't be root dir!"
        exit 1
    fi
    $remote_device_run $remote_device_name shell "rm -rf $remote_device_work_dir"
    $remote_device_run $remote_device_name shell "mkdir -p $remote_device_work_dir"
}

function android_cpu_build_target() {
    local os=$1
    local arch=$2
    local toolchain=$3

    # Build all of tests
    rm -rf $BUILD_DIRECTORY
    mkdir -p $BUILD_DIRECTORY
    cd $BUILD_DIRECTORY
    prepare_workspace $ROOT_DIR $BUILD_DIRECTORY

    cmake .. \
        -DWITH_GPU=OFF \
        -DWITH_MKL=OFF \
        -DLITE_WITH_CUDA=OFF \
        -DLITE_WITH_X86=OFF \
        -DLITE_WITH_ARM=ON \
        -DWITH_ARM_DOTPROD=ON \
        -DWITH_TESTING=ON \
        -DLITE_BUILD_EXTRA=ON \
        -DLITE_WITH_TRAIN=ON \
        -DARM_TARGET_OS=$os -DARM_TARGET_ARCH_ABI=$arch -DARM_TARGET_LANG=$toolchain
    make lite_compile_deps -j$NUM_CORES_FOR_COMPILE
}

function android_cpu_build_and_test() {
    build_and_test_on_remote_device $OS_LIST $ARCH_LIST $TOOLCHAIN_LIST $UNIT_TEST_CHECK_LIST $UNIT_TEST_FILTER_TYPE android_cpu_build_target android_cpu_prepare_device $REMOTE_DEVICE_TYPE $REMOTE_DEVICE_LIST $REMOTE_DEVICE_WORK_DIR
}

# ARMLinux (RK3399/pro, Raspberry pi etc.)
function armlinux_cpu_prepare_device() {
    local os=$1
    local arch=$2
    local toolchain=$3
    local remote_device_name=$4
    local remote_device_work_dir=$5
    local remote_device_check=$6
    local remote_device_run=$7

    # Check device is available
    $remote_device_check $remote_device_name
    if [[ $? -ne 0 ]]; then
        echo "$remote_device_name not found!"
        exit 1
    fi

    # Create work dir on the remote device
    if [[ -z "$remote_device_work_dir" ]]; then
        echo "$remote_device_work_dir can't be empty!"
        exit 1
    fi
    if [[ "$remote_device_work_dir" == "/" ]]; then
        echo "$remote_device_work_dir can't be root dir!"
        exit 1
    fi
    $remote_device_run $remote_device_name shell "rm -rf $remote_device_work_dir"
    $remote_device_run $remote_device_name shell "mkdir -p $remote_device_work_dir"
}

function armlinux_cpu_build_target() {
    local os=$1
    local arch=$2
    local toolchain=$3

    # Build all of tests
    rm -rf $BUILD_DIRECTORY
    mkdir -p $BUILD_DIRECTORY
    cd $BUILD_DIRECTORY
    prepare_workspace $ROOT_DIR $BUILD_DIRECTORY

    cmake .. \
        -DWITH_GPU=OFF \
        -DWITH_MKL=OFF \
        -DLITE_WITH_CUDA=OFF \
        -DLITE_WITH_X86=OFF \
        -DLITE_WITH_ARM=ON \
        -DWITH_ARM_DOTPROD=ON \
        -DWITH_TESTING=ON \
        -DLITE_BUILD_EXTRA=ON \
        -DLITE_WITH_TRAIN=ON \
        -DARM_TARGET_OS=$os -DARM_TARGET_ARCH_ABI=$arch -DARM_TARGET_LANG=$toolchain
    make lite_compile_deps -j$NUM_CORES_FOR_COMPILE
}

function armlinux_cpu_build_and_test() {
    build_and_test_on_remote_device $OS_LIST $ARCH_LIST $TOOLCHAIN_LIST $UNIT_TEST_CHECK_LIST $UNIT_TEST_FILTER_TYPE armlinux_cpu_build_target armlinux_cpu_prepare_device $REMOTE_DEVICE_TYPE $REMOTE_DEVICE_LIST $REMOTE_DEVICE_WORK_DIR
}

# Huawei Kirin NPU
function huawei_kirin_npu_prepare_device() {
    local os=$1
    local arch=$2
    local toolchain=$3
    local remote_device_name=$4
    local remote_device_work_dir=$5
    local remote_device_check=$6
    local remote_device_run=$7
    local sdk_root_dir=$8

    # Check device is available
    $remote_device_check $remote_device_name
    if [[ $? -ne 0 ]]; then
        echo "$remote_device_name not found!"
        exit 1
    fi

    # Create work dir on the remote device
    if [[ -z "$remote_device_work_dir" ]]; then
        echo "$remote_device_work_dir can't be empty!"
        exit 1
    fi
    if [[ "$remote_device_work_dir" == "/" ]]; then
        echo "$remote_device_work_dir can't be root dir!"
        exit 1
    fi
    $remote_device_run $remote_device_name shell "rm -rf $remote_device_work_dir"
    $remote_device_run $remote_device_name shell "mkdir -p $remote_device_work_dir"

    # Only root user can use HiAI runtime libraries in the android shell executables
    $remote_device_run $remote_device_name root
    if [[ $? -ne 0 ]]; then
        echo "$remote_device_name hasn't the root permission!"
        exit 1
    fi

    # Copy the runtime libraries of HiAI DDK to the target device
    local sdk_lib_dir=""
    if [[ $arch == "armv8" ]]; then
        sdk_lib_dir="$sdk_root_dir/lib64"
    elif [[ $arch == "armv7" ]]; then
        sdk_lib_dir="$sdk_root_dir/lib"
    else
        echo "$arch isn't supported by HiAI DDK!"
        exit 1
    fi
    $remote_device_run $remote_device_name push "$sdk_lib_dir/*" "$remote_device_work_dir"
}

function huawei_kirin_npu_build_target() {
    local os=$1
    local arch=$2
    local toolchain=$3
    local sdk_root_dir=$4

    # Build all of tests
    rm -rf $BUILD_DIRECTORY
    mkdir -p $BUILD_DIRECTORY
    cd $BUILD_DIRECTORY
    prepare_workspace $ROOT_DIR $BUILD_DIRECTORY

    cmake .. \
        -DWITH_GPU=OFF \
        -DWITH_MKL=OFF \
        -DLITE_WITH_CUDA=OFF \
        -DLITE_WITH_X86=OFF \
        -DLITE_WITH_ARM=ON \
        -DWITH_ARM_DOTPROD=ON \
        -DWITH_TESTING=ON \
        -DLITE_BUILD_EXTRA=ON \
        -DLITE_WITH_TRAIN=ON \
        -DANDROID_STL_TYPE="c++_shared" \
        -DLITE_WITH_NPU=ON \
        -DNPU_DDK_ROOT="$sdk_root_dir" \
        -DARM_TARGET_OS=$os -DARM_TARGET_ARCH_ABI=$arch -DARM_TARGET_LANG=$toolchain
    make lite_compile_deps -j$NUM_CORES_FOR_COMPILE
}

function huawei_kirin_npu_build_and_test() {
    build_and_test_on_remote_device $OS_LIST $ARCH_LIST $TOOLCHAIN_LIST $UNIT_TEST_CHECK_LIST $UNIT_TEST_FILTER_TYPE huawei_kirin_npu_build_target huawei_kirin_npu_prepare_device $REMOTE_DEVICE_TYPE $REMOTE_DEVICE_LIST $REMOTE_DEVICE_WORK_DIR "$(readlink -f ./hiai_ddk_lib_330)"
}

# Rockchip NPU
function rockchip_npu_prepare_device() {
    local os=$1
    local arch=$2
    local toolchain=$3
    local remote_device_name=$4
    local remote_device_work_dir=$5
    local remote_device_check=$6
    local remote_device_run=$7
    local sdk_root_dir=$8

    # Check device is available
    $remote_device_check $remote_device_name
    if [[ $? -ne 0 ]]; then
        echo "$remote_device_name not found!"
        exit 1
    fi

    # Create work dir on the remote device
    if [[ -z "$remote_device_work_dir" ]]; then
        echo "$remote_device_work_dir can't be empty!"
        exit 1
    fi
    if [[ "$remote_device_work_dir" == "/" ]]; then
        echo "$remote_device_work_dir can't be root dir!"
        exit 1
    fi
    $remote_device_run $remote_device_name shell "rm -rf $remote_device_work_dir"
    $remote_device_run $remote_device_name shell "mkdir -p $remote_device_work_dir"

    # Copy the runtime libraries of Rockchip NPU to the target device
    local sdk_lib_dir=""
    if [[ $arch == "armv8" ]]; then
        sdk_lib_dir="$sdk_root_dir/lib64"
    elif [[ $arch == "armv7hf" ]]; then
        sdk_lib_dir="$sdk_root_dir/lib"
    else
        echo "$arch isn't supported by Rockchip NPU SDK!"
        exit 1
    fi
    $remote_device_run $remote_device_name push "$sdk_lib_dir/librknpu_ddk.so" "$remote_device_work_dir"
}

function baidu_xpu_build_and_test() {
    local unit_test_check_list=$2
    local unit_test_filter_type=$3
    local sdk_url=$4
    local sdk_env=$5
    local xdnn_url=$6
    local xre_url=$7

    # Build all of unittests and model tests
    cur_dir=$(pwd)
    BUILD_DIRECTORY=$cur_dir/build.lite.xpu.test

    rm -rf $BUILD_DIRECTORY
    mkdir -p $BUILD_DIRECTORY
    cd $BUILD_DIRECTORY
    prepare_workspace $ROOT_DIR $BUILD_DIRECTORY

    cmake .. \
        -DWITH_PYTHON=OFF \
        -DWITH_TESTING=ON \
        -DLITE_WITH_ARM=OFF \
        -DWITH_GPU=OFF \
        -DWITH_MKLDNN=OFF \
        -DLITE_WITH_X86=ON \
        -DWITH_MKL=ON \
        -DLITE_BUILD_EXTRA=ON \
        -DLITE_WITH_XPU=ON \
        -DLITE_WITH_LTO=OFF \
        -DXPU_SDK_URL=$sdk_url \
        -DXPU_XDNN_URL=$xdnn_url \
        -DXPU_XRE_URL=$xre_url \
        -DXPU_SDK_ENV=$sdk_env \
        -DXPU_SDK_ROOT=$XPU_SDK_ROOT

    make lite_compile_deps -j$NUM_CORES_FOR_COMPILE

    # Run all of unittests and model tests
    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$PWD/third_party/install/mklml/lib"
    export GLOG_v=$UNIT_TEST_LOG_LEVEL
    local unit_test_check_items=(${unit_test_check_list//,/ })
    for test_name in $(cat $TESTS_FILE); do
        local is_matched=0
        for unit_test_check_item in ${unit_test_check_items[@]}; do
            if [[ "$unit_test_check_item" == "$test_name" ]]; then
                echo "$test_name on the checklist."
                is_matched=1
                break
            fi
        done
        # black list
        if [[ $is_matched -eq 1 && $unit_test_filter_type -eq 0 ]]; then
            continue
        fi
        # white list
        if [[ $is_matched -eq 0 && $unit_test_filter_type -eq 1 ]]; then
            continue
        fi
        ctest -V -R ^$test_name$
    done
}

function main() {
    # Parse command line.
    for i in "$@"; do
        case $i in
        --os_list=*)
            OS_LIST="${i#*=}"
            shift
            ;;
        --arch_list=*)
            ARCH_LIST="${i#*=}"
            shift
            ;;
        --toolchain_list=*)
            TOOLCHAIN_LIST="${i#*=}"
            shift
            ;;
        --unit_test_check_list=*)
            UNIT_TEST_CHECK_LIST="${i#*=}"
            shift
            ;;
        --unit_test_filter_type=*)
            UNIT_TEST_FILTER_TYPE="${i#*=}"
            shift
            ;;
        --unit_test_log_level=*)
            UNIT_TEST_LOG_LEVEL="${i#*=}"
            shift
            ;;
        --remote_device_type=*)
            REMOTE_DEVICE_TYPE="${i#*=}"
            shift
            ;;
        --remote_device_list=*)
            REMOTE_DEVICE_LIST="${i#*=}"
            shift
            ;;
        --remote_device_work_dir=*)
            REMOTE_DEVICE_WORK_DIR="${i#*=}"
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
        --xpu_xdnn_url=*)
            XPU_XDNN_URL="${i#*=}"
            shift
            ;;
        --xpu_xre_url=*)
            XPU_XRE_URL="${i#*=}"
            shift
            ;;
        --xpu_sdk_root=*)
            XPU_SDK_ROOT="${i#*=}"
            shift
            ;;
        android_cpu_build_and_test)
            android_cpu_build_and_test
            shift
            ;;
        armlinux_cpu_build_and_test)
            armlinux_cpu_build_and_test
            shift
            ;;
        huawei_kirin_npu_build_and_test)
            huawei_kirin_npu_build_and_test
            shift
            ;;
        huawei_ascend_npu_build_and_test)
            huawei_ascend_npu_build_and_test
            shift
            ;;
        rockchip_npu_build_and_test)
            rockchip_npu_build_and_test
            shift
            ;;
        baidu_xpu_disable_xtcl_build_and_test)
            baidu_xpu_build_and_test OFF $UNIT_TEST_CHECK_LIST $UNIT_TEST_FILTER_TYPE $XPU_SDK_URL $XPU_SDK_ENV $XPU_XDNN_URL $XPU_XRE_URL
            shift
            ;;
        baidu_xpu_enable_xtcl_build_and_test)
            baidu_xpu_build_and_test ON $UNIT_TEST_CHECK_LIST $UNIT_TEST_FILTER_TYPE $XPU_SDK_URL $XPU_SDK_ENV $XPU_XDNN_URL $XPU_XRE_URL
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
