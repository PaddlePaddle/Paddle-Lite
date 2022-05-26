#!/bin/bash
set +x
set -e

#####################################################################################################
# 1. global variables, you can change them according to your requirements
#####################################################################################################
# only support armv8|x86.
ARCH=armv8
# ON or OFF, default OFF.

CMAKE_EXTRA_OPTIONS=""
BUILD_EXTRA=OFF
BUILD_PYTHON=OFF
BUILD_DIR=$(pwd)
OPTMODEL_DIR=""
BUILD_TAILOR=OFF
BUILD_CV=OFF
WITH_LOG=ON
WITH_MKL=ON
WITH_METAL=OFF
WITH_OPENCL=OFF
LITE_ON_TINY_PUBLISH=ON
WITH_STATIC_MKL=OFF
WITH_AVX=ON
WITH_EXCEPTION=OFF
WITH_LIGHT_WEIGHT_FRAMEWORK=OFF
WITH_PROFILE=OFF
WITH_PRECISION_PROFILE=OFF
WITH_BENCHMARK=OFF
WITH_LTO=OFF
WITH_TESTING=OFF
BUILD_ARM82_FP16=OFF
BUILD_ARM82_INT8_SDOT=OFF
PYTHON_EXECUTABLE_OPTION=""
PY_VERSION=""
workspace=$PWD/$(dirname $0)/../../
OPTMODEL_DIR=""
IOS_DEPLOYMENT_TARGET=11.0
# num of threads used during compiling..
readonly NUM_PROC=${LITE_BUILD_THREADS:-4}
#####################################################################################################


#####################################################################################################
# 2. local variables, these variables should not be changed.
#####################################################################################################
# url that stores third-party tar.gz file to accelerate third-party lib installation
readonly THIRDPARTY_URL=https://paddlelite-data.bj.bcebos.com/third_party_libs/
readonly THIRDPARTY_TAR=third-party-91a9ab3.tar.gz

# on mac environment, we should expand the maximum file num to compile successfully
os_name=`uname -s`
if [ ${os_name} == "Darwin" ]; then
   ulimit -n 1024
fi
#####################################################################################################

####################################################################################################
# 3. compiling functions
####################################################################################################
function prepare_workspace {
    local root_dir=$1
    local build_dir=$2
    # in build directory
    # 1. Prepare gen_code file
    GEN_CODE_PATH_PREFIX=$build_dir/lite/gen_code
    mkdir -p ${GEN_CODE_PATH_PREFIX}
    touch ${GEN_CODE_PATH_PREFIX}/__generated_code__.cc
    # 2.Prepare debug tool
    DEBUG_TOOL_PATH_PREFIX=$build_dir/lite/tools/debug
    mkdir -p ${DEBUG_TOOL_PATH_PREFIX}
    cp $root_dir/lite/tools/debug/analysis_tool.py ${DEBUG_TOOL_PATH_PREFIX}/
}


function prepare_opencl_source_code {
    local root_dir=$1
    local build_dir=$2
    # in build directory
    # Prepare opencl_kernels_source.cc file
    GEN_CODE_PATH_OPENCL=$root_dir/lite/backends/opencl
    rm -f GEN_CODE_PATH_OPENCL/opencl_kernels_source.cc
    OPENCL_KERNELS_PATH=$root_dir/lite/backends/opencl/cl_kernel
    mkdir -p ${GEN_CODE_PATH_OPENCL}
    touch $GEN_CODE_PATH_OPENCL/opencl_kernels_source.cc
    python $root_dir/lite/tools/cmake_tools/gen_opencl_code.py $OPENCL_KERNELS_PATH $GEN_CODE_PATH_OPENCL/opencl_kernels_source.cc
}

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
}

# function of set options for benchmark
function set_benchmark_options {
  BUILD_EXTRA=ON
  WITH_EXCEPTION=ON
  LITE_ON_TINY_PUBLISH=OFF

  if [ ${WITH_PROFILE} == "ON" ] || [ ${WITH_PRECISION_PROFILE} == "ON" ]; then
    WITH_LOG=ON
  else
    WITH_LOG=OFF
  fi
}

function build_opt {
    cd $workspace
    prepare_thirdparty
    mkdir -p build.opt
    cd build.opt
    opt_arch=$(echo `uname -a` | awk -F " " '{print $15}')
    with_x86=OFF
    if [ $opt_arch == "arm64" ]; then
       with_x86=OFF
    else
       with_x86=ON
    fi
    cmake .. -DWITH_LITE=ON \
      -DLITE_ON_MODEL_OPTIMIZE_TOOL=ON \
      -DWITH_TESTING=OFF \
      -DLITE_BUILD_EXTRA=ON \
      -DLITE_WITH_X86=${with_x86} \
      -DWITH_MKL=OFF
    make opt -j$NUM_PROC
}

function make_armosx {
    prepare_thirdparty
    if [ "${BUILD_PYTHON}" == "ON" ]; then
      BUILD_EXTRA=ON
      LITE_ON_TINY_PUBLISH=OFF
    fi
    local arch=armv8
    local os=armmacos
    if [ "${WITH_STRIP}" == "ON" ]; then
        BUILD_EXTRA=ON
    fi

    if [ "${WITH_BENCHMARK}" == "ON" ]; then
        set_benchmark_options
    fi

    build_dir=$workspace/build.macos.${os}.${arch}
    if [ ${WITH_METAL} == "ON" ]; then
      BUILD_EXTRA=ON
      build_dir=${build_dir}.metal
    fi

    if [ ${WITH_OPENCL} == "ON" ]; then
        build_dir=${build_dir}.opencl
        prepare_opencl_source_code $workspace
    fi

    if [ ${WITH_TESTING} == "ON" ]; then
      BUILD_EXTRA=ON
      LITE_ON_TINY_PUBLISH=OFF
    fi

    if [ "${BUILD_ARM82_FP16}" == "ON" ]; then
      TOOLCHAIN=clang
    fi

    if [ -d $build_dir ]
    then
        rm -rf $build_dir
    fi
    if [ ! -d third-party ]; then
        git checkout third-party
    fi

    echo "building arm macos target into $build_dir"
    echo "target arch: $arch"
    mkdir -p ${build_dir}
    cd ${build_dir}
    GEN_CODE_PATH_PREFIX=lite/gen_code
    mkdir -p ./${GEN_CODE_PATH_PREFIX}
    touch ./${GEN_CODE_PATH_PREFIX}/__generated_code__.cc
    cmake $workspace \
            -DWITH_LITE=ON \
            -DWITH_TESTING=${WITH_TESTING} \
            -DLITE_WITH_ARM=ON \
            -DLITE_WITH_METAL=${WITH_METAL} \
            -DLITE_WITH_OPENCL=${WITH_OPENCL} \
            -DLITE_ON_TINY_PUBLISH=${LITE_ON_TINY_PUBLISH} \
            -DLITE_WITH_PROFILE=${WITH_PROFILE} \
            -DLITE_WITH_LIGHT_WEIGHT_FRAMEWORK=ON \
            -DLITE_WITH_PRECISION_PROFILE=${WITH_PRECISION_PROFILE} \
            -DLITE_WITH_OPENMP=OFF \
            -DWITH_ARM_DOTPROD=ON \
            -DLITE_WITH_X86=OFF \
            -DLITE_WITH_M1=ON \
            -DLITE_WITH_PYTHON=${BUILD_PYTHON} \
            -DPY_VERSION=$PY_VERSION \
            -DLITE_WITH_LOG=$WITH_LOG \
            -DLITE_WITH_EXCEPTION=$WITH_EXCEPTION \
            -DLITE_BUILD_TAILOR=$BUILD_TAILOR \
            -DLITE_OPTMODEL_DIR=$OPTMODEL_DIR \
            -DARM_TARGET_ARCH_ABI=$arch \
            -DLITE_BUILD_EXTRA=$BUILD_EXTRA \
            -DLITE_WITH_CV=$BUILD_CV \
            -DLITE_WITH_ARM82_FP16=$BUILD_ARM82_FP16 \
            -DDEPLOYMENT_TARGET=${IOS_DEPLOYMENT_TARGET} \
            -DLITE_WITH_LIGHT_WEIGHT_FRAMEWORK=ON \
            -DARM_TARGET_OS=armmacos
    if [ "${WITH_BENCHMARK}" == "ON" ]; then
        make benchmark_bin -j$NUM_PROC
    elif [ "${WITH_TESTING}" == "ON" ]; then
        make lite_compile_deps -j$NUM_PROC
    else
        make publish_inference -j$NUM_PROC
    fi
    cd -
}

function make_x86 {
  prepare_thirdparty

  root_dir=$(pwd)
  build_directory=$BUILD_DIR/build.lite.x86

  if [ "${WITH_BENCHMARK}" == "ON" ]; then
    set_benchmark_options
  fi

  if [ ${WITH_OPENCL} == "ON" ]; then
    BUILD_EXTRA=ON
    build_directory=$BUILD_DIR/build.lite.x86.opencl
    prepare_opencl_source_code $root_dir $build_directory
  fi

  if [ ${WITH_METAL} == "ON" ]; then
    BUILD_EXTRA=ON
    build_directory=${build_directory}.metal
  fi

  if [ ${WITH_TESTING} == "ON" ]; then
    BUILD_EXTRA=ON
    LITE_ON_TINY_PUBLISH=OFF
  fi

  if [ ${BUILD_PYTHON} == "ON" ]; then
    BUILD_EXTRA=ON
    LITE_ON_TINY_PUBLISH=OFF
  fi

  if [ ! -d third-party ]; then
    git checkout third-party
  fi

  if [ -d $build_directory ]
  then
    rm -rf $build_directory
  fi
  mkdir -p $build_directory
  cd $build_directory

  prepare_workspace $root_dir $build_directory

  cmake $root_dir  -DWITH_MKL=${WITH_MKL}  \
            -DWITH_STATIC_MKL=${WITH_STATIC_MKL}  \
            -DWITH_TESTING=${WITH_TESTING} \
            -DWITH_AVX=${WITH_AVX} \
            -DWITH_MKLDNN=OFF    \
            -DLITE_WITH_X86=ON  \
            -DWITH_LITE=ON \
            -DLITE_WITH_LIGHT_WEIGHT_FRAMEWORK=${WITH_LIGHT_WEIGHT_FRAMEWORK} \
            -DLITE_ON_TINY_PUBLISH=OFF \
            -DLITE_WITH_PROFILE=${WITH_PROFILE} \
            -DLITE_WITH_PRECISION_PROFILE=${WITH_PRECISION_PROFILE} \
            -DLITE_WITH_ARM=OFF \
            -DLITE_WITH_METAL=${WITH_METAL} \
            -DLITE_WITH_OPENCL=${WITH_OPENCL} \
            -DWITH_GPU=OFF \
            -DLITE_WITH_PYTHON=${BUILD_PYTHON} \
            -DLITE_BUILD_EXTRA=${BUILD_EXTRA} \
            -DLITE_BUILD_TAILOR=${BUILD_TAILOR} \
            -DLITE_OPTMODEL_DIR=${OPTMODEL_DIR} \
            -DLITE_WITH_LOG=${WITH_LOG} \
            -DLITE_WITH_EXCEPTION=$WITH_EXCEPTION \
            -DLITE_WITH_LTO=${WITH_LTO} \
            -DCMAKE_BUILD_TYPE=Release \
            -DPY_VERSION=$PY_VERSION \
            $PYTHON_EXECUTABLE_OPTION

  if [ "${WITH_BENCHMARK}" == "ON" ]; then
    make benchmark_bin -j$NUM_PROC
  elif [ "${WITH_TESTING}" == "ON" ]; then
    make lite_compile_deps -j$NUM_PROC
  else
    make publish_inference -j$NUM_PROC
  fi
  cd -
}

function print_usage {
    echo "----------------------------------------------------------------------------------------------------------------------------------------"
    echo -e "| Methods of compiling Padddle-Lite Arm Macos library:                                                                                 |"
    echo "----------------------------------------------------------------------------------------------------------------------------------------"
    echo -e "|  compile macos armv8 library:                                                                                                        |"
    echo -e "|     ./lite/tools/build_macos.sh                                                                                                      |"
    echo -e "|  print help information:                                                                                                             |"
    echo -e "|     ./lite/tools/build_macos.sh help                                                                                                 |"
    echo -e "|                                                                                                                                      |"
    echo -e "|  for arm macos:                                                                                                                      |"
    echo -e "|  optional argument:                                                                                                                  |"
    echo -e "|     --with_metal: (OFF|ON); controls whether to build with Metal, default is OFF                                                     |"
    echo -e "|     --with_cv: (OFF|ON); controls whether to compile cv functions into lib, default is OFF                                           |"
    echo -e "|     --with_log: (OFF|ON); controls whether to print log information, default is ON                                                   |"
    echo -e "|     --with_exception: (OFF|ON); controls whether to throw the exception when error occurs, default is OFF                            |"
    echo -e "|     --with_extra: (OFF|ON); controls whether to publish extra operators and kernels for (sequence-related model such as OCR or NLP) |"
    echo -e "|     --with_benchmark: (OFF|ON); controls whether to compile benchmark binary, default is OFF                                         |"
    echo -e "|     --with_testing: (OFF|ON); controls whether to compile unit test, default is OFF                                                  |"
    echo -e "|     --with_arm82_fp16: (OFF|ON); controls whether to include FP16 kernels, default is OFF                                            |"
    echo -e "|                                  warning: when --with_arm82_fp16=ON, toolchain will be set as clang, arch will be set as armv8.      |"
    echo -e "|                                                                                                                                      |"
    echo -e "|  compiling for macos OPT tool:                                                                              |"
    echo -e "|     ./lite/tools/build_macos.sh build_optimize_tool                                                                              |"
    echo -e "|  arguments of benchmark binary compiling for macos x86:                                                                              |"
    echo -e "|     ./lite/tools/build_macos.sh --with_benchmark=ON x86                                                                              |"
    echo -e "|                                                                                                                                      |"
    echo -e "|  arguments of benchmark binary compiling for macos opencl(only support --gpu_precision=fp32):                                        |"
    echo -e "|     ./lite/tools/build_macos.sh --with_benchmark=ON arm64                                                                            |"
    echo -e "|                                                                                                                                      |"
    echo -e "|  arguments of striping lib according to input model:(armv8, gcc, c++_static)                                                         |"
    echo -e "|     ./lite/tools/build_macos.sh --with_strip=ON --opt_model_dir=YourOptimizedModelDir                                                |"
    echo -e "|     --with_strip: (OFF|ON); controls whether to strip lib accrding to input model, default is OFF                                    |"
    echo -e "|     --opt_model_dir: (absolute path to optimized model dir) required when compiling striped library                                  |"
    echo -e "|  detailed information about striping lib:  https://paddle-lite.readthedocs.io/zh/latest/user_guides/library_tailoring.html           |"
    echo "----------------------------------------------------------------------------------------------------------------------------------------"

}

function main {
    if [ -z "$1" ]; then
        make_osx $ARCH
        exit 0
    fi

    # Parse command line.
    for i in "$@"; do
        case $i in
            --with_metal=*)
                WITH_METAL="${i#*=}"
                shift
                ;;
            --with_opencl=*)
                WITH_OPENCL="${i#*=}"
                shift
                ;;
            --with_extra=*)
                BUILD_EXTRA="${i#*=}"
                shift
                ;;
            --with_cv=*)
                BUILD_CV="${i#*=}"
                shift
                ;;
            --with_python=*)
                BUILD_PYTHON="${i#*=}"
                shift
                ;;
            --build_dir=*)
                BUILD_DIR="${i#*=}"
                shift
                ;;
            --opt_model_dir=*)
                OPTMODEL_DIR="${i#*=}"
                OPTMODEL_DIR=$(readlinkf $OPTMODEL_DIR)
                shift
                ;;
            --build_tailor=*)
                BUILD_TAILOR="${i#*=}"
                shift
                ;;
            --with_log=*)
                WITH_LOG="${i#*=}"
                shift
                ;;
            # controls whether to include FP16 kernels, default is OFF
            --with_arm82_fp16=*)
                BUILD_ARM82_FP16="${i#*=}"
                shift
                ;;
            --with_mkl=*)
                WITH_MKL="${i#*=}"
                shift
                ;;
            --with_static_mkl=*)
                WITH_STATIC_MKL="${i#*=}"
                shift
                ;;
            --with_avx=*)
                WITH_AVX="${i#*=}"
                shift
                ;;
            --with_exception=*)
                WITH_EXCEPTION="${i#*=}"
                if [[ $WITH_EXCEPTION == "ON" && $ARM_OS=="android" && $ARM_ABI == "armv7" && $ARM_LANG != "clang" ]]; then
                     set +x
                     echo
                     echo -e "error: only clang provide C++ exception handling support for 32-bit ARM."
                     echo
                     exit 1
                fi
                shift
                ;;
            --with_profile=*)
                WITH_PROFILE="${i#*=}"
                shift
                ;;
            --with_precision_profile=*)
                WITH_PRECISION_PROFILE="${i#*=}"
                shift
                ;;
            --with_benchmark=*)
                WITH_BENCHMARK="${i#*=}"
                shift
                ;;
            --with_lto=*)
                WITH_LTO="${i#*=}"
                shift
                ;;
            --tiny_publish=*)
                LITE_ON_TINY_PUBLISH="${i#*=}"
                shift
                ;;
            --python_executable=*)
                PYTHON_EXECUTABLE_OPTION="-DPYTHON_EXECUTABLE=${i#*=}"
                shift
                ;;
            --python_version=*)
                PY_VERSION="${i#*=}"
                shift
                ;;
            --ios_deployment_target=*)
                IOS_DEPLOYMENT_TARGET="${i#*=}"
                shift
                ;;
            arm64)
               make_armosx
               shift
               ;;
            x86)
               make_x86
               shift
               ;;
            build_optimize_tool)
                build_opt
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
    exit 0
}

main $@
