#!/bin/bash
set -ex

#####################################################################################################
# 1. global variables, you can change them according to your requirements
#####################################################################################################
# armv7 or armv8, default armv8.
ARM_ABI=armv8
# ON or OFF, default OFF.
BUILD_EXTRA=OFF
# controls whether to compile cv functions into lib, default is OFF.
BUILD_CV=OFF
# controls whether to hide log information, default is ON.
SHUTDOWN_LOG=ON
BUILD_DIR=$(pwd)
# options of striping lib according to input model.
OPTMODEL_DIR=""
BUILD_TAILOR=OFF
# num of threads used during compiling..
readonly NUM_PROC=${LITE_BUILD_THREADS:-4}
#####################################################################################################


#####################################################################################################
# 2. local variables, these variables should not be changed.
#####################################################################################################
# on mac environment, we should expand the maximum file num to compile successfully
os_nmae=`uname -s`
if [ ${os_nmae} == "Darwin" ]; then
   ulimit -n 1024
fi
#####################################################################################################

####################################################################################################
# 4. compiling functions
####################################################################################################
function make_ios {
    local abi=$1

    if [ ${abi} == "armv8" ]; then
        local os=ios64
    elif [ ${abi} == "armv7" ]; then
        local os=ios
    else
        echo -e "Error: unsupported arm_abi: ${abi} \t --arm_abi: armv8|armv7"
        exit 1
    fi

    build_dir=build.ios.${os}.${abi}
    echo "building ios target into $build_dir"
    echo  then"target os: $os"
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
            -DLITE_BUILD_TAILOR=$BUILD_TAILOR \
            -DLITE_OPTMODEL_DIR=$OPTMODEL_DIR \
            -DLITE_WITH_LIGHT_WEIGHT_FRAMEWORK=ON \
            -DARM_TARGET_ARCH_ABI=$abi \
            -DLITE_BUILD_EXTRA=$BUILD_EXTRA \
            -DLITE_WITH_CV=$BUILD_CV \
            -DARM_TARGET_OS=$os

    make publish_inference -j$NUM_PROC
    cd -
}


function print_usage {
    set +x
    echo -e "\nMethods of compiling Paddle-Lite iOS library:"
    echo "----------------------------------------"
    echo -e "compile ios armv8 lib:"
    echo -e "   ./build_ios.sh --arm_abi=armv8"
    echo -e "compile ios armv7 lib:"
    echo -e "   ./build_ios.sh --arm_abi=armv7"
    echo
    echo -e "optional arguments:"
    echo -e "--arm_abi:\t armv8|armv7, required."
    echo -e "--shutdown_log: (OFF|ON); controls whether to shutdown log, default is ON"
    echo -e "--build_cv: (OFF|ON); controls whether to compile cv functions into lib, default is OFF"
    echo -e "--shutdown_log: (OFF|ON); controls whether to hide log information, default is ON"
    echo -e "--build_extra: (OFF|ON); controls whether to publish extra operators and kernels for (sequence-related model such as OCR or NLP)"
    echo
    echo -e "arguments of striping lib according to input model:"
    echo -e "--build_strip: (OFF|ON); controls whether to strip lib accrding to input model, default is OFF"
    echo -e "--opt_model_dir: (path to optimized model dir); contains absolute path to optimized model dir"
    echo "----------------------------------------"
    echo
}

function main {
    if [ -z "$1" ]; then
        print_usage
        exit -1
    fi

    echo $1
    # Parse command line.
    for i in "$@"; do
        case $i in
            --arm_abi=*)
                ARM_ABI="${i#*=}"
                make_ios $ARM_ABI
                shift
                ;;
            --build_extra=*)
                BUILD_EXTRA="${i#*=}"
                shift
                ;;
            --build_cv=*)
                BUILD_CV="${i#*=}"
                shift
                ;;
            --build_dir=*)
                BUILD_DIR="${i#*=}"
                shift
		;;
            --opt_model_dir=*)
                OPTMODEL_DIR="${i#*=}"
                shift
                ;;
            --build_tailor=*)
                BUILD_TAILOR="${i#*=}"
                shift
                ;;
            --shutdown_log=*)
                SHUTDOWN_LOG="${i#*=}"
                shift
                ;;
            ios)
                make_ios $ARM_ABI
                shift
                ;;
            *)
                # unknown option
                print_usage
                exit 1
                ;;
        esac
        make_ios $ARM_ABI
    done
}

main $@
