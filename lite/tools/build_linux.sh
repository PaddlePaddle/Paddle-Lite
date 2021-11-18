#!/bin/bash
set -e

#####################################################################################################
# 1. global variables, you can change them according to your requirements
#####################################################################################################
# armv8 or armv7hf or armv7, default armv8.
ARCH=armv8
# gcc or clang, default gcc.
TOOLCHAIN=gcc
# ON or OFF, default OFF.
WITH_EXTRA=OFF
# controls whether to compile python lib, default is OFF.
WITH_PYTHON=OFF
PY_VERSION=""
# ON or OFF, default is OFF
WITH_STATIC_LIB=OFF
# controls whether to compile cv functions into lib, default is OFF.
WITH_CV=OFF
# controls whether to print log information, default is ON.
WITH_LOG=ON
# controls whether to throw the exception when error occurs, default is OFF
WITH_EXCEPTION=OFF
# options of striping lib according to input model.
WITH_STRIP=OFF
OPTMODEL_DIR=""
# options of compiling x86 lib
WITH_STATIC_MKL=OFF
WITH_AVX=ON
# options of compiling OPENCL lib.
WITH_OPENCL=OFF
# options of compiling Metal lib for Mac OS.
WITH_METAL=OFF
# options of compiling rockchip NPU lib.
WITH_ROCKCHIP_NPU=OFF
ROCKCHIP_NPU_SDK_ROOT="$(pwd)/rknpu_ddk"  # Download RKNPU SDK from https://github.com/airockchip/rknpu_ddk.git
# options of compiling NNAdapter lib
WITH_NNADAPTER=OFF
NNADAPTER_WITH_ROCKCHIP_NPU=OFF
NNADAPTER_ROCKCHIP_NPU_SDK_ROOT="$(pwd)/rknpu_ddk"  # Download RKNPU SDK from https://github.com/airockchip/rknpu_ddk.git
NNADAPTER_WITH_IMAGINATION_NNA=OFF
NNADAPTER_IMAGINATION_NNA_SDK_ROOT="$(pwd)/imagination_nna_sdk"
NNADAPTER_WITH_HUAWEI_ASCEND_NPU=OFF
NNADAPTER_HUAWEI_ASCEND_NPU_SDK_ROOT="/usr/local/Ascend/ascend-toolkit/latest"
NNADAPTER_WITH_AMLOGIC_NPU=OFF
NNADAPTER_AMLOGIC_NPU_SDK_ROOT="$(pwd)/amlnpu_ddk"
NNADAPTER_WITH_CAMBRICON_MLU=OFF
NNADAPTER_CAMBRICON_MLU_SDK_ROOT="$(pwd)/cambricon_mlu_sdk"
# options of compiling baidu XPU lib.
WITH_BAIDU_XPU=OFF
WITH_BAIDU_XPU_XTCL=OFF
BAIDU_XPU_SDK_ROOT=""
BAIDU_XPU_SDK_URL=""
BAIDU_XPU_SDK_ENV=""
# options of compiling intel fpga.
WITH_INTEL_FPGA=OFF
INTEL_FPGA_SDK_ROOT="$(pwd)/intel_fpga_sdk"
# options of adding training ops
WITH_TRAIN=OFF
# options of building tiny publish so
WITH_TINY_PUBLISH=ON
# controls whether to include FP16 kernels, default is OFF
BUILD_ARM82_FP16=OFF
# options of profiling
WITH_PROFILE=OFF
WITH_PRECISION_PROFILE=OFF
# option of benchmark, default is OFF
WITH_BENCHMARK=OFF
# option of light weight framework, default is OFF
WITH_LIGHT_WEIGHT_FRAMEWORK=OFF
# num of threads used during compiling..
readonly NUM_PROC=${LITE_BUILD_THREADS:-4}
#####################################################################################################




#####################################################################################################
# 2. local variables, these variables should not be changed.
#####################################################################################################
# url that stores third-party tar.gz file to accelerate third-party lib installation
readonly THIRDPARTY_URL=https://paddlelite-data.bj.bcebos.com/third_party_libs/
readonly THIRDPARTY_TAR=third-party-801f670.tar.gz

# absolute path of Paddle-Lite.
readonly workspace=$PWD/$(dirname $0)/../../
# basic options for linux compiling.
readonly CMAKE_COMMON_OPTIONS="-DWITH_LITE=ON \
                            -DCMAKE_BUILD_TYPE=Release \
                            -DWITH_MKLDNN=OFF \
                            -DWITH_TESTING=OFF"

# function of set options for benchmark
function set_benchmark_options {
  WITH_EXTRA=ON
  WITH_EXCEPTION=ON
  WITH_NNADAPTER=ON
  if [ "${ARCH}" == "x86" ]; then
    # Turn off opencl. Additional third party library need to be installed on
    # Linux. Otherwise opencl is not supported on Linux. See link for more info:
    # https://software.intel.com/content/www/us/en/develop/articles/opencl-drivers.html
    WITH_OPENCL=OFF
    WITH_LIGHT_WEIGHT_FRAMEWORK=OFF
  else
    WITH_LIGHT_WEIGHT_FRAMEWORK=ON
    WITH_OPENCL=ON
  fi
  if [ ${WITH_PROFILE} == "ON" ] || [ ${WITH_PRECISION_PROFILE} == "ON" ]; then
    WITH_LOG=ON
  else
    WITH_LOG=OFF
  fi
}

# mutable options for linux compiling.
function init_cmake_mutable_options {
    if [ "$WITH_PYTHON" = "ON" -a "$WITH_TINY_PUBLISH" = "ON" ]; then
        echo "Warning: build full_publish to use python."
        WITH_TINY_PUBLISH=OFF
    fi
    if [ "$WITH_TRAIN" = "ON" -a "$WITH_TINY_PUBLISH" = "ON" ]; then
        echo "Warning: build full_publish to add training ops."
        WITH_TINY_PUBLISH=OFF
    fi

    if [ "$BUILD_TAILOR" = "ON" -a "$OPTMODEL_DIR" = "" ]; then
        echo "Error: set OPTMODEL_DIR if BUILD_TAILOR is ON."
        exit 1
    fi

    if [ "${ARCH}" == "x86" ]; then
        with_x86=ON
        arm_target_os=""
        WITH_LIGHT_WEIGHT_FRAMEWORK=OFF
        WITH_TINY_PUBLISH=OFF
    else
        with_arm=ON
        arm_arch=$ARCH
        arm_target_os=armlinux
        WITH_LIGHT_WEIGHT_FRAMEWORK=ON
        WITH_AVX=OFF
    fi

    if [ "${WITH_STRIP}" == "ON" ]; then
        WITH_EXTRA=ON
    fi

    if [ "${WITH_BAIDU_XPU}" == "ON" ]; then
        WITH_EXTRA=ON
        WITH_TINY_PUBLISH=OFF
    fi

    if [ "${WITH_BENCHMARK}" == "ON" ]; then
        set_benchmark_options
    fi

    cmake_mutable_options="-DLITE_WITH_ARM=$with_arm \
                        -DLITE_WITH_X86=$with_x86 \
                        -DARM_TARGET_ARCH_ABI=$arm_arch \
                        -DARM_TARGET_OS=$arm_target_os \
                        -DARM_TARGET_LANG=$TOOLCHAIN \
                        -DLITE_WITH_LIGHT_WEIGHT_FRAMEWORK=$WITH_LIGHT_WEIGHT_FRAMEWORK \
                        -DLITE_BUILD_EXTRA=$WITH_EXTRA \
                        -DLITE_WITH_PYTHON=$WITH_PYTHON \
                        -DPY_VERSION=$PY_VERSION \
                        -DLITE_WITH_STATIC_LIB=$WITH_STATIC_LIB \
                        -DLITE_WITH_CV=$WITH_CV \
                        -DLITE_WITH_LOG=$WITH_LOG \
                        -DLITE_WITH_EXCEPTION=$WITH_EXCEPTION \
                        -DLITE_BUILD_TAILOR=$WITH_STRIP \
                        -DLITE_OPTMODEL_DIR=$OPTMODEL_DIR \
                        -DWITH_STATIC_MKL=$WITH_STATIC_MKL \
                        -DWITH_AVX=$WITH_AVX \
                        -DLITE_WITH_OPENCL=$WITH_OPENCL \
                        -DLITE_WITH_METAL=$WITH_METAL \
                        -DLITE_WITH_RKNPU=$WITH_ROCKCHIP_NPU \
                        -DRKNPU_DDK_ROOT=$ROCKCHIP_NPU_SDK_ROOT \
                        -DLITE_WITH_XPU=$WITH_BAIDU_XPU \
                        -DLITE_WITH_XTCL=$WITH_BAIDU_XPU_XTCL \
                        -DXPU_SDK_ROOT=$BAIDU_XPU_SDK_ROOT \
                        -DXPU_SDK_URL=$BAIDU_XPU_SDK_URL \
                        -DXPU_SDK_ENV=$BAIDU_XPU_SDK_ENV \
                        -DLITE_WITH_TRAIN=$WITH_TRAIN  \
                        -DLITE_WITH_NNADAPTER=$WITH_NNADAPTER \
                        -DNNADAPTER_WITH_ROCKCHIP_NPU=$NNADAPTER_WITH_ROCKCHIP_NPU \
                        -DNNADAPTER_ROCKCHIP_NPU_SDK_ROOT=$NNADAPTER_ROCKCHIP_NPU_SDK_ROOT \
                        -DNNADAPTER_WITH_IMAGINATION_NNA=$NNADAPTER_WITH_IMAGINATION_NNA \
                        -DNNADAPTER_IMAGINATION_NNA_SDK_ROOT=$NNADAPTER_IMAGINATION_NNA_SDK_ROOT \
                        -DNNADAPTER_WITH_HUAWEI_ASCEND_NPU=$NNADAPTER_WITH_HUAWEI_ASCEND_NPU \
                        -DNNADAPTER_HUAWEI_ASCEND_NPU_SDK_ROOT=$NNADAPTER_HUAWEI_ASCEND_NPU_SDK_ROOT \
                        -DNNADAPTER_WITH_AMLOGIC_NPU=$NNADAPTER_WITH_AMLOGIC_NPU \
                        -DNNADAPTER_AMLOGIC_NPU_SDK_ROOT=$NNADAPTER_AMLOGIC_NPU_SDK_ROOT \
                        -DNNADAPTER_WITH_CAMBRICON_MLU=$NNADAPTER_WITH_CAMBRICON_MLU \
                        -DNNADAPTER_CAMBRICON_MLU_SDK_ROOT=$NNADAPTER_CAMBRICON_MLU_SDK_ROOT \
                        -DLITE_WITH_INTEL_FPGA=$WITH_INTEL_FPGA \
                        -DINTEL_FPGA_SDK_ROOT=${INTEL_FPGA_SDK_ROOT} \
                        -DLITE_WITH_PROFILE=${WITH_PROFILE} \
                        -DLITE_WITH_ARM82_FP16=$BUILD_ARM82_FP16 \
                        -DLITE_WITH_PRECISION_PROFILE=${WITH_PRECISION_PROFILE} \
                        -DLITE_ON_TINY_PUBLISH=$WITH_TINY_PUBLISH"

}
#####################################################################################################





####################################################################################################
# 3. functions of prepare workspace before compiling
####################################################################################################

# 3.1 generate `__generated_code__.cc`, which is dependended by some targets in cmake.
# here we fake an empty file to make cmake works.
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


# 3.2 prepare source code of opencl lib
# here we bundle all cl files into a cc file to bundle all opencl kernels into a single lib
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

# 3.3 prepare third_party libraries for compiling
# here we store third_party libraries into Paddle-Lite/third-party
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
####################################################################################################





####################################################################################################
# 4. compiling functions
####################################################################################################

# 4.1 function of publish compiling
# here we only compile light_api lib
function make_publish_so {
    init_cmake_mutable_options

    if [ "$WITH_TINY_PUBLISH" = "OFF" ]; then
        prepare_thirdparty
    else
        if [ ! -d third-party ] ; then
            git checkout third-party
        fi
    fi

    build_dir=$workspace/build.lite.linux.$ARCH.$TOOLCHAIN
    if [ "${WITH_OPENCL}" = "ON" ]; then
        build_dir=${build_dir}.opencl
    fi
    if [ "${WITH_METAL}" = "ON" ]; then
        build_dir=${build_dir}.metal
    fi
    if [ "${WITH_BAIDU_XPU}" = "ON" ]; then
        build_dir=${build_dir}.baidu_xpu
    fi

    if [ -d $build_dir ]; then
        rm -rf $build_dir
    fi
    mkdir -p $build_dir
    cd $build_dir

    prepare_workspace $workspace $build_dir

    if [ "${WITH_OPENCL}" = "ON" ]; then
        prepare_opencl_source_code $workspace $build_dir
    fi

    cmake $workspace \
        ${CMAKE_COMMON_OPTIONS} \
        ${cmake_mutable_options}

    if [ "${WITH_BENCHMARK}" == "ON" ]; then
        make benchmark_bin -j$NUM_PROC
    else
        make publish_inference -j$NUM_PROC
    fi
    cd - > /dev/null
}
####################################################################################################



function print_usage {
    echo "--------------------------------------------------------------------------------------------------------------------------------------------------------"
    echo -e "| Methods of compiling Padddle-Lite Linux library:                                                                                                     |"
    echo "--------------------------------------------------------------------------------------------------------------------------------------------------------"
    echo -e "|  compile linux library: (armv8, gcc)                                                                                                                 |"
    echo -e "|     ./lite/tools/build_linux.sh                                                                                                                      |"
    echo -e "|  print help information:                                                                                                                             |"
    echo -e "|     ./lite/tools/build_linux.sh help                                                                                                                 |"
    echo -e "|                                                                                                                                                      |"
    echo -e "|  optional argument:                                                                                                                                  |"
    echo -e "|     --arch: (armv8|armv7hf|armv7|x86), default is armv8                                                                                              |"
    echo -e "|     --toolchain: (gcc|clang), defalut is gcc                                                                                                         |"
    echo -e "|     --with_extra: (OFF|ON); controls whether to publish extra operators and kernels for (sequence-related model such as OCR or NLP), default is OFF  |"
    echo -e "|     --with_python: (OFF|ON); controls whether to build python lib or whl, default is OFF                                                             |"
    echo -e "|     --python_version: (2.7|3.5|3.7); controls python version to compile whl, default is None                                                         |"
    echo -e "|     --with_static_lib: (OFF|ON); controls whether to publish c++ api static lib, default is OFF                                                      |"
    echo -e "|     --with_cv: (OFF|ON); controls whether to compile cv functions into lib, default is OFF                                                           |"
    echo -e "|     --with_log: (OFF|ON); controls whether to print log information, default is ON                                                                   |"
    echo -e "|     --with_exception: (OFF|ON); controls whether to throw the exception when error occurs, default is OFF                                            |"
    echo -e "|     --with_profile: (OFF|ON); controls whether to profile speed, default is OFF                                                                      |"
    echo -e "|     --with_precision_profile: (OFF|ON); controls whether to profile precision, default is OFF                                                        |"
    echo -e "|     --with_benchmark: (OFF|ON); controls whether to compile benchmark binary, default is OFF                                                         |"
    echo -e "|                                                                                                                                                      |"
    echo -e "|  arguments of benchmark binary compiling:                                                                                                            |"
    echo -e "|     ./lite/tools/build_linux.sh --with_benchmark=ON full_publish                                                                                     |"
    echo -e "|                                                                                                                                                      |"
    echo -e "|  arguments of striping lib according to input model:                                                                                                 |"
    echo -e "|     ./lite/tools/build_linux.sh --with_strip=ON --opt_model_dir=YourOptimizedModelDir                                                                |"
    echo -e "|     --with_strip: (OFF|ON); controls whether to strip lib accrding to input model, default is OFF                                                    |"
    echo -e "|     --opt_model_dir: (absolute path to optimized model dir) required when compiling striped library                                                  |"
    echo -e "|  detailed information about striping lib:  https://paddle-lite.readthedocs.io/zh/latest/user_guides/library_tailoring.html                           |"
    echo -e "|                                                                                                                                                      |"
    echo -e "|  arguments of x86 library compiling:                                                                                                                 |"
    echo -e "|     ./lite/tools/build_linux.sh --arch=x86                                                                                                           |"
    echo -e "|     --with_static_mkl: (OFF|ON); controls whether to compile static mkl lib, default is OFF                                                          |"
    echo -e "|     --with_avx: (OFF|ON); controls whether to use avx , default is ON                                                                                |"
    echo -e "|                                                                                                                                                      |"
    echo -e "|  arguments of opencl library compiling:                                                                                                              |"
    echo -e "|     ./lite/tools/build_linux.sh --with_opencl=ON                                                                                                     |"
    echo -e "|     --with_opencl: (OFF|ON); controls whether to compile lib for opencl, default is OFF                                                              |"
    echo -e "|                                                                                                                                                      |"
    echo -e "|  arguments of rockchip npu library compiling:                                                                                                        |"
    echo -e "|     ./lite/tools/build_linux.sh --with_rockchip_npu=ON --rockchip_npu_sdk_root=YourRockchipNpuSdkPath                                                |"
    echo -e "|     --with_rockchip_npu: (OFF|ON); controls whether to compile lib for rockchip_npu, default is OFF                                                  |"
    echo -e "|     --rockchip_npu_sdk_root: (path to rockchip_npu DDK file) required when compiling rockchip_npu library                                            |"
    echo -e "|             you can download rockchip NPU SDK from:  https://github.com/airockchip/rknpu_ddk.git                                                     |"
    echo -e "|  detailed information about Paddle-Lite RKNPU:  https://paddle-lite.readthedocs.io/zh/latest/demo_guides/rockchip_npu.html                           |"
    echo -e "|                                                                                                                                                      |"
    echo -e "|  arguments of cambricon mlu library compiling:                                                                                                       |"
    echo -e "|     ./lite/tools/build_linux.sh --with_cambricon_mlu=ON --cambricon_mlu_sdk_root=YourCambriconMluSdkPath                                             |"
    echo -e "|     --with_cambricon_mlu: (OFF|ON); controls whether to compile lib for cambricon_mlu, default is OFF                                                |"
    echo -e "|     --cambricon_mlu_sdk_root: (path to cambricon_mlu SDK file) required when compiling cambricon_mlu library                                         |"
    echo -e "|             you can download cambricon MLU SDK from:                                                                                                 |"
    echo -e "|  detailed information about Paddle-Lite CAMBRICON MLU:  https://paddle-lite.readthedocs.io/zh/latest/demo_guides/cambricon_mlu.html                  |"
    echo -e "|                                                                                                                                                      |"
    echo -e "|  arguments of baidu xpu library compiling:                                                                                                           |"
    echo -e "|     ./lite/tools/build_linux.sh --arch=x86 --with_baidu_xpu=ON                                                                                       |"
    echo -e "|     ./lite/tools/build_linux.sh --arch=armv8 --with_baidu_xpu=ON                                                                                     |"
    echo -e "|     --with_baidu_xpu: (OFF|ON); controls whether to compile lib for baidu_xpu, default is OFF.                                                       |"
    echo -e "|     --with_baidu_xpu_xtcl: (OFF|ON); controls whether to enable xtcl for baidu_xpu, default is OFF.                                                  |"
    echo -e "|     --baidu_xpu_sdk_root: (path to baidu_xpu DDK file) optional, default is None                                                                     |"
    echo -e "|     --baidu_xpu_sdk_url: (baidu_xpu sdk download url) optional, default is 'https://baidu-kunlun-product.cdn.bcebos.com/KL-SDK/klsdk-dev_paddle'     |"
    echo -e "|     --baidu_xpu_sdk_env: (bdcentos_x86_64|centos7_x86_64|ubuntu_x86_64|kylin_aarch64) optional,                                                      |"
    echo -e "|             default is bdcentos_x86_64(if x86) / kylin_aarch64(if arm)                                                                               |"
    echo "--------------------------------------------------------------------------------------------------------------------------------------------------------"
    echo
}

function main {
    # Parse command line.
    for i in "$@"; do
        case $i in
            # armv8 or armv7hf or armv7, default armv8
            --arch=*)
                ARCH="${i#*=}"
                shift
                ;;
            # gcc or clang, default gcc
            --toolchain=*)
                TOOLCHAIN="${i#*=}"
                shift
                ;;
            # ON or OFF, default OFF
            --with_extra=*)
                WITH_EXTRA="${i#*=}"
                shift
                ;;
            # controls whether to compile cplus static library, default is OFF
            --with_static_lib=*)
                WITH_STATIC_LIB="${i#*=}"
                shift
                ;;
            # ON or OFF, default OFF
            --with_python=*)
                WITH_PYTHON="${i#*=}"
                shift
                ;;
            # 2.7 or 3.5 or 3.7, default is None
            --python_version=*)
                PY_VERSION="${i#*=}"
                shift
                ;;
            # ON or OFF, default OFF
            --with_cv=*)
                WITH_CV="${i#*=}"
                shift
                ;;
            # ON or OFF, default ON
            --with_log=*)
                WITH_LOG="${i#*=}"
                shift
                ;;
            # ON or OFF, default OFF
            --with_exception=*)
                WITH_EXCEPTION="${i#*=}"
                shift
                ;;
            # ON or OFF, default OFF
            --with_strip=*)
                BUILD_TAILOR="${i#*=}"
                shift
                ;;
            # string, absolute path to optimized model dir
            --opt_model_dir=*)
                OPTMODEL_DIR="${i#*=}"
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
            # compiling lib which can operate on opencl and cpu.
            --with_opencl=*)
                WITH_OPENCL="${i#*=}"
                shift
                ;;
            # compiling lib for Mac OS with GPU support.
            --with_metal=*)
                WITH_METAL="${i#*=}"
                shift
                ;;
            # compiling lib which can operate on rockchip npu.
            --with_rockchip_npu=*)
                WITH_ROCKCHIP_NPU="${i#*=}"
                shift
                ;;
            --rockchip_npu_sdk_root=*)
                ROCKCHIP_NPU_SDK_ROOT="${i#*=}"
                shift
                ;;
            # compiling lib which can operate on nnadapter.
            --with_nnadapter=*)
                WITH_NNADAPTER="${i#*=}"
                shift
                ;;
            --nnadapter_with_rockchip_npu=*)
                NNADAPTER_WITH_ROCKCHIP_NPU="${i#*=}"
                shift
                ;;
            --nnadapter_rockchip_npu_sdk_root=*)
                NNADAPTER_ROCKCHIP_NPU_SDK_ROOT="${i#*=}"
                shift
                ;;
             --nnadapter_with_imagination_nna=*)
                NNADAPTER_WITH_IMAGINATION_NNA="${i#*=}"
                shift
                ;;
            --nnadapter_imagination_nna_sdk_root=*)
                NNADAPTER_IMAGINATION_NNA_SDK_ROOT="${i#*=}"
                shift
                ;;
             --nnadapter_with_huawei_ascend_npu=*)
                NNADAPTER_WITH_HUAWEI_ASCEND_NPU="${i#*=}"
                shift
                ;;
            --nnadapter_huawei_ascend_npu_sdk_root=*)
                NNADAPTER_HUAWEI_ASCEND_NPU_SDK_ROOT="${i#*=}"
                shift
                ;;
            --nnadapter_with_amlogic_npu=*)
                NNADAPTER_WITH_AMLOGIC_NPU="${i#*=}"
                shift
                ;;
            --nnadapter_amlogic_npu_sdk_root=*)
                NNADAPTER_AMLOGIC_NPU_SDK_ROOT="${i#*=}"
                shift
                ;;
            --nnadapter_with_cambricon_mlu=*)
                NNADAPTER_WITH_CAMBRICON_MLU="${i#*=}"
                shift
                ;;
            --nnadapter_cambricon_mlu_sdk_root=*)
                NNADAPTER_CAMBRICON_MLU_SDK_ROOT="${i#*=}"
                shift
                ;;
            # compiling lib which can operate on baidu xpu.
            --with_baidu_xpu=*)
                WITH_BAIDU_XPU="${i#*=}"
                shift
                ;;
            --with_baidu_xpu_xtcl=*)
                WITH_BAIDU_XPU_XTCL="${i#*=}"
                shift
                ;;
            --baidu_xpu_sdk_root=*)
                BAIDU_XPU_SDK_ROOT="${i#*=}"
                if [ -n "${BAIDU_XPU_SDK_ROOT}" ]; then
                    BAIDU_XPU_SDK_ROOT=$(readlink -f ${BAIDU_XPU_SDK_ROOT})
                fi
                shift
                ;;
            --baidu_xpu_sdk_url=*)
                BAIDU_XPU_SDK_URL="${i#*=}"
                shift
                ;;
            --baidu_xpu_sdk_env=*)
                BAIDU_XPU_SDK_ENV="${i#*=}"
                shift
                ;;
            # compiling lib which can operate on intel fpga.
            --with_intel_fpga=*)
                WITH_INTEL_FPGA="${i#*=}"
                shift
                ;;
            --intel_fpga_sdk_root=*)
                INTEL_FPGA_SDK_ROOT="${i#*=}"
                shift
                ;;
            # controls whether to include FP16 kernels, default is OFF
            --with_arm82_fp16=*)
                BUILD_ARM82_FP16="${i#*=}"
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
            # compiling lib with benchmark feature, default OFF.
            --with_benchmark=*)
                WITH_BENCHMARK="${i#*=}"
                shift
                ;;
            # ON or OFF, default OFF
            --with_train=*)
                WITH_TRAIN="${i#*=}"
                shift
                ;;
            # compiling result contains both light_api and cxx_api lib.
            full_publish)
                WITH_TINY_PUBLISH=OFF
                make_publish_so
                exit 0
                ;;
            # print help info
            help)
                print_usage
                exit 0
                ;;
            # unknown option
            *)
                echo "Error: unsupported argument \"${i#*=}\""
                print_usage
                exit 1
                ;;
        esac
    done

    make_publish_so
}

main $@
