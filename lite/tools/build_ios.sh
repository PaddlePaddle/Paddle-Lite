#!/bin/bash
set +x

#####################################################################################################
# 1. global variables, you can change them according to your requirements
#####################################################################################################
# armv7 or armv8, default armv8.
ARCH=armv8
# ON or OFF, default OFF.
WITH_EXTRA=OFF
# controls whether to compile cv functions into lib, default is OFF.
WITH_CV=OFF
# controls whether to hide log information, default is ON.
WITH_LOG=ON
# controls whether to throw the exception when error occurs, default is OFF 
WITH_EXCEPTION=OFF
# absolute path of Paddle-Lite.
workspace=$PWD/$(dirname $0)/../../
# options of striping lib according to input model.
OPTMODEL_DIR=""
WITH_STRIP=OFF
IOS_DEPLOYMENT_TARGET=9.0
# num of threads used during compiling..
readonly NUM_PROC=${LITE_BUILD_THREADS:-4}
#####################################################################################################


#####################################################################################################
# 2. local variables, these variables should not be changed.
#####################################################################################################
# on mac environment, we should expand the maximum file num to compile successfully
os_name=`uname -s`
if [ ${os_name} == "Darwin" ]; then
   ulimit -n 1024
fi
#####################################################################################################

####################################################################################################
# 3. compiling functions
####################################################################################################
function make_ios {
    local arch=$1

    if [ ${arch} == "armv8" ]; then
        local os=ios64
    elif [ ${arch} == "armv7" ]; then
        local os=ios
    else
        echo -e "Error: unsupported arch: ${arch} \t --arch: armv8|armv7"
        exit 1
    fi

    if [ "${WITH_STRIP}" == "ON" ]; then
        WITH_EXTRA=ON
    fi

    build_dir=$workspace/build.ios.${os}.${arch}
    if [ -d $build_dir ]
    then
        rm -rf $build_dir
    fi
    echo "building ios target into $build_dir"
    echo "target arch: $arch"
    mkdir -p ${build_dir}
    cd ${build_dir}
    GEN_CODE_PATH_PREFIX=lite/gen_code
    mkdir -p ./${GEN_CODE_PATH_PREFIX}
    touch ./${GEN_CODE_PATH_PREFIX}/__generated_code__.cc
    cmake $workspace \
            -DWITH_LITE=ON \
            -DLITE_WITH_ARM=ON \
            -DLITE_ON_TINY_PUBLISH=ON \
            -DLITE_WITH_OPENMP=OFF \
            -DWITH_ARM_DOTPROD=OFF \
            -DLITE_WITH_LIGHT_WEIGHT_FRAMEWORK=ON \
            -DLITE_WITH_X86=OFF \
            -DLITE_WITH_LOG=$WITH_LOG \
            -DLITE_WITH_EXCEPTION=$WITH_EXCEPTION \
            -DLITE_BUILD_TAILOR=$WITH_STRIP \
            -DLITE_OPTMODEL_DIR=$OPTMODEL_DIR \
            -DARM_TARGET_ARCH_ABI=$arch \
            -DLITE_BUILD_EXTRA=$WITH_EXTRA \
            -DLITE_WITH_CV=$WITH_CV \
            -DDEPLOYMENT_TARGET=${IOS_DEPLOYMENT_TARGET} \
            -DARM_TARGET_OS=$os

    make publish_inference -j$NUM_PROC
    cd -
}


function print_usage {
    echo "----------------------------------------------------------------------------------------------------------------------------------------"
    echo -e "| Methods of compiling Padddle-Lite iOS library:                                                                                       |"
    echo "----------------------------------------------------------------------------------------------------------------------------------------"
    echo -e "|  compile iOS armv8 library:                                                                                                          |"
    echo -e "|     ./lite/tools/build_ios.sh                                                                                                        |"
    echo -e "|  compile iOS armv7 library:                                                                                                          |"
    echo -e "|     ./lite/tools/build_ios.sh  --arch=armv7                                                                                          |"
    echo -e "|  print help information:                                                                                                             |"
    echo -e "|     ./lite/tools/build_ios.sh help                                                                                                   |"
    echo -e "|                                                                                                                                      |"
    echo -e "|  optional argument:                                                                                                                  |"
    echo -e "|     --arch: (armv8|armv7), default is armv8                                                                                          |"
    echo -e "|     --with_cv: (OFF|ON); controls whether to compile cv functions into lib, default is OFF                                           |"
    echo -e "|     --with_log: (OFF|ON); controls whether to print log information, default is ON                                                   |"
    echo -e "|     --with_exception: (OFF|ON); controls whether to throw the exception when error occurs, default is OFF                            |"
    echo -e "|     --with_extra: (OFF|ON); controls whether to publish extra operators and kernels for (sequence-related model such as OCR or NLP)  |"
    echo -e "|     --ios_deployment_target: (default: 9.0); Set the minimum compatible system version for ios deployment.                           |"
    echo -e "|                                                                                                                                      |"
    echo -e "|  arguments of striping lib according to input model:(armv8, gcc, c++_static)                                                         |"
    echo -e "|     ./lite/tools/build_android.sh --with_strip=ON --opt_model_dir=YourOptimizedModelDir                                              |"
    echo -e "|     --with_strip: (OFF|ON); controls whether to strip lib accrding to input model, default is OFF                                    |"
    echo -e "|     --opt_model_dir: (absolute path to optimized model dir) required when compiling striped library                                  |"
    echo -e "|  detailed information about striping lib:  https://paddle-lite.readthedocs.io/zh/latest/user_guides/library_tailoring.html           |"
    echo "----------------------------------------------------------------------------------------------------------------------------------------"

}

function main {
    if [ -z "$1" ]; then
        make_ios $ARCH
        exit -1
    fi

    # Parse command line.
    for i in "$@"; do
        case $i in
            --arch=*)
                ARCH="${i#*=}"
                shift
                ;;
            --with_extra=*)
                WITH_EXTRA="${i#*=}"
                shift
                ;;
            --with_cv=*)
                WITH_CV="${i#*=}"
                shift
                ;;
            --opt_model_dir=*)
                OPTMODEL_DIR="${i#*=}"
                shift
                ;;
            --with_strip=*)
                WITH_STRIP="${i#*=}"
                shift
                ;;
            --with_log=*)
                WITH_LOG="${i#*=}"
                shift
                ;;
            --with_exception=*)
                WITH_EXCEPTION="${i#*=}"
                shift
                ;;
            --ios_deployment_target=*)
                IOS_DEPLOYMENT_TARGET="${i#*=}"
                shift
                ;;
            help)
                print_usage
                exit 0
                ;;
            *)
                # unknown option
                print_usage
                exit 1
                ;;
        esac
    done
    make_ios $ARCH
    exit 0
}

main $@
