#!/bin/bash
set -e

function print_usage {
    set +x
    echo -e "\nUSAGE:"
    echo "----------------------------------------"
    echo -e "compile ios tiny publish so lib:"
    echo -e "   ./build_ios.sh --arm_os=<os> --arm_abi=<abi>"
    echo -e "argument choices:"
    echo -e "--arm_os:\t ios|ios64, default is ios64"
    echo -e "--arm_abi:\t armv7|armv8, default is armv8"
    echo "----------------------------------------"
    echo
}

function build_ios {
    local os=$1
    local abi=$2
    build_dir=build.ios.${os}.${abi}
    echo "building ios target into $build_dir"
    echo "target os: $os"
    echo "target abi: $abi"
    mkdir -p ${build_dir}
    cd ${build_dir}
    GEN_CODE_PATH_PREFIX=lite/gen_code
    mkdir -p ./${GEN_CODE_PATH_PREFIX}
    touch ./${GEN_CODE_PATH_PREFIX}/__generated_code__.cc

    cmake .. \
            -DWITH_GPU=OFF \
            -DWITH_MKL=OFF \
            -DWITH_LITE=ON \
            -DLITE_WITH_CUDA=OFF \
            -DLITE_WITH_X86=OFF \
            -DLITE_WITH_ARM=ON \
            -DWITH_TESTING=OFF \
            -DLITE_WITH_JAVA=OFF \
            -DLITE_SHUTDOWN_LOG=ON \
            -DLITE_ON_TINY_PUBLISH=ON \
            -DLITE_WITH_OPENMP=OFF \
            -DWITH_ARM_DOTPROD=OFF \
            -DLITE_WITH_LIGHT_WEIGHT_FRAMEWORK=ON \
            -DARM_TARGET_ARCH_ABI=$abi \
            -DARM_TARGET_OS=$os

    make -j4 publish_inference
    cd -
}

function main {
    ARM_OS="ios64"
    ARM_ABI="armv8"
    # Parse command line.
    for i in "$@"; do
        case $i in
            --arm_os=*)
                ARM_OS="${i#*=}"
                shift
                ;;
            --arm_abi=*)
                ARM_ABI="${i#*=}"
                shift
                ;;
            *)
                # unknown option
                print_usage
                exit 1
                ;;
        esac
    done
    build_ios $ARM_OS $ARM_ABI
}

main $@
