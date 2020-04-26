#!/bin/bash
set +x

# 1. using functions and variables defined in basicfuncs library file
source basicfuncs

#####################################################################################################
# 2. local variables, these variables should not be changed.
#####################################################################################################
# basic options for android compiling.
readonly CMAKE_COMMON_OPTIONS="-DWITH_LITE=ON \
                               -DLITE_WITH_ARM=ON \
                               -DLITE_WITH_LIGHT_WEIGHT_FRAMEWORK=ON \
                               -DLITE_WITH_X86=OFF \
                               -DWITH_TESTING=OFF \
                               -DARM_TARGET_OS=android"

####################################################################################################
# 3. compiling functions
####################################################################################################

# 3.1 function of tiny_publish compiling
# here we only compile light_api lib
function make_tiny_publish_so {
  build_dir=$workspace/build.lite.android.$ARM_ABI.$TOOLCHAIN
  if [ "${WITH_OPENCL}" == "ON" ]; then
      build_dir=${build_dir}.opencl
  fi
  if [ "${WITH_npu}" == "ON" ]; then
      build_dir=${build_dir}.npu
  fi


  if [ -d $build_dir ]
  then
      rm -rf $build_dir
  fi
  mkdir -p $build_dir
  cd $build_dir

  if [ "${WITH_OPENCL}" == "ON" ]; then
      prepare_opencl_source_code $workspace $build_dir
  fi


  local cmake_mutable_options="
      -DLITE_BUILD_EXTRA=$WITH_EXTRA \
      -DLITE_SHUTDOWN_LOG=$SHUTDOWN_LOG \
      -DLITE_BUILD_TAILOR=$WITH_STRIP \
      -DLITE_OPTMODEL_DIR=$OPTMODEL_DIR \
      -DLITE_WITH_JAVA=$WITH_JAVA \
      -DLITE_WITH_CV=$WITH_CV \
      -DLITE_WITH_NPU=$WITH_HUAWEI_KIRIN_NPU \
      -DNPU_DDK_ROOT=$HUAWEI_KIRIN_NPU_SDK_ROOT \
      -DLITE_WITH_OPENCL=$WITH_OPENCL \
      -DARM_TARGET_ARCH_ABI=$ARM_ABI \
      -DARM_TARGET_LANG=$TOOLCHAIN \
      -DANDROID_STL_TYPE=$ANDROID_STL"

  cmake $workspace \
      ${CMAKE_COMMON_OPTIONS} \
      ${cmake_mutable_options}  \
      -DLITE_ON_TINY_PUBLISH=ON 

  # todo: third_party of opencl should be moved into git submodule and cmake later
  if [ "${WITH_OPENCL}" == "ON" ]; then
      make opencl_clhpp -j$NUM_PROC 
  fi

  make publish_inference -j$NUM_PROC
  cd - > /dev/null
}

# 3.2 function of full_publish compiling
# here we compile both light_api lib and full_api lib
function make_full_publish_so {

  prepare_thirdparty

  build_directory=$workspace/build.lite.android.$ARM_ABI.$ARM_LANG

  if [ -d $build_directory ]
  then
      rm -rf $build_directory
  fi
  mkdir -p $build_directory
  cd $build_directory

  prepare_workspace $workspace $build_directory

  if [ "${WITH_OPENCL}" == "ON" ]; then
      prepare_opencl_source_code $workspace $build_dir
  fi

  local cmake_mutable_options="
      -DLITE_BUILD_EXTRA=$WITH_EXTRA \
      -DLITE_SHUTDOWN_LOG=$SHUTDOWN_LOG \
      -DLITE_BUILD_TAILOR=$WITH_STRIP \
      -DLITE_OPTMODEL_DIR=$OPTMODEL_DIR \
      -DLITE_WITH_JAVA=$WITH_JAVA \
      -DLITE_WITH_CV=$WITH_CV \
      -DLITE_WITH_NPU=$WITH_HUAWEI_KIRIN_NPU \
      -DNPU_DDK_ROOT=$HUAWEI_KIRIN_NPU_SDK_ROOT \
      -DLITE_WITH_OPENCL=$WITH_OPENCL \
      -DARM_TARGET_ARCH_ABI=$ARM_ABI \
      -DARM_TARGET_LANG=$ARM_LANG \
      -DLITE_WITH_TRAIN=$WITH_TRAIN \
      -DANDROID_STL_TYPE=$ANDROID_STL"

  cmake $workspace \
      ${CMAKE_COMMON_OPTIONS} \
      ${cmake_mutable_options}

  # todo: third_party of opencl should be moved into git submodule and cmake later
  if [ "${WITH_OPENCL}" == "ON" ]; then
      make opencl_clhpp -j$NUM_PROC
  fi

  make publish_inference -j$NUM_PROC
  cd - > /dev/null
}


# 3.3 function of print help information
function print_usage {
    echo "----------------------------------------------------------------------------------------------------------------------------------------"
    echo -e "| Methods of compiling Padddle-Lite Android library:                                                                                   |"
    echo "----------------------------------------------------------------------------------------------------------------------------------------"
    echo -e "|  compile android library: (armv8, gcc, c++_static)                                                                                   |"
    echo -e "|     ./lite/tools/build_android.sh                                                                                                    |"
    echo -e "|  print help information:                                                                                                             |"
    echo -e "|     ./lite/tools/build_android.sh help                                                                                               |"
    echo -e "|                                                                                                                                      |"
    echo -e "|  optional argument:                                                                                                                  |"
    echo -e "|     --arm_abi: (armv8|armv7), default is armv8                                                                                       |"
    echo -e "|     --toolchain: (gcc|clang), defalut is gcc                                                                                         |"
    echo -e "|     --android_stl: (c++_static|c++_shared|gnu_static|gnu_shared), default is c++_static                                              |"
    echo -e "|     --with_java: (OFF|ON); controls whether to publish java api lib, default is ON                                                   |"
    echo -e "|     --with_cv: (OFF|ON); controls whether to compile cv functions into lib, default is OFF                                           |"
    echo -e "|     --shutdown_log: (OFF|ON); controls whether to hide log information, default is ON                                                |"
    echo -e "|     --with_extra: (OFF|ON); controls whether to publish extra operators and kernels for (sequence-related model such as OCR or NLP)  |"
    echo -e "|                                                                                                                                      |"
    echo -e "|  arguments of striping lib according to input model:(armv8, gcc, c++_static)                                                         |"
    echo -e "|     ./lite/tools/build_android.sh --with_strip=ON --opt_model_dir=YourOptimizedModelDir                                              |"
    echo -e "|     --with_strip: (OFF|ON); controls whether to strip lib accrding to input model, default is OFF                                    |"
    echo -e "|     --opt_model_dir: (absolute path to optimized model dir) required when compiling striped library                                  |"
    echo -e "|  detailed information about striping lib:  https://paddle-lite.readthedocs.io/zh/latest/user_guides/library_tailoring.html           |"
    echo -e "|                                                                                                                                      |"
    echo -e "|  arguments of npu library compiling:(armv8, gcc, c++_static)                                                                         |"
    echo -e "|     ./lite/tools/build_android.sh --with_huawei_kirin_npu=ON --huawei_kirin_npu_sdk_root=YourNpuSdkPath                              |"
    echo -e "|     --with_huawei_kirin_npu: (OFF|ON); controls whether to compile lib for huawei_kirin_npu, default is OFF                          |"
    echo -e "|     --huawei_kirin_npu_sdk_root: (path to huawei HiAi DDK file) required when compiling npu library                                  |"
    echo -e "|             you can download huawei HiAi DDK from:  https://developer.huawei.com/consumer/cn/hiai/                                   |"
    echo -e "|  detailed information about Paddle-Lite NPU:  https://paddle-lite.readthedocs.io/zh/latest/demo_guides/npu.html                      |"
    echo -e "|                                                                                                                                      |"
    echo -e "|  arguments of opencl library compiling:(armv8, gcc, c++_static)                                                                      |"
    echo -e "|     ./lite/tools/build_android.sh --with_opencl=ON                                                                                   |"
    echo -e "|     --with_opencl: (OFF|ON); controls whether to compile lib for opencl, default is OFF                                              |"
    echo "----------------------------------------------------------------------------------------------------------------------------------------"
    echo
}

####################################################################################################


####################################################################################################
# 4. main functions: choose compiling method according to input argument
####################################################################################################
function main {
    if [ -z "$1" ]; then
        # compiling result contains light_api lib only, recommanded.
        make_tiny_publish_so $ARM_ABI $TOOLCHAIN $ANDROID_STL
    fi

    # Parse command line.
    for i in "$@"; do
        case $i in
            # armv7 or armv8, default armv8
            --arm_abi=*)
                ARM_ABI="${i#*=}"
                shift
                ;;
            # gcc or clang, default gcc
            --toolchain=*)
                TOOLCHAIN="${i#*=}"
                shift
                ;;
            # c++_static or c++_shared, default c++_static
            --android_stl=*)
                ANDROID_STL="${i#*=}"
                shift
                ;;
            # ON or OFF, default OFF
            --with_extra=*)
                WITH_EXTRA="${i#*=}"
                shift
                ;;
            # ON or OFF, default OFF
            --with_cv=*)
                WITH_CV="${i#*=}"
                shift
                ;;
            # ON or OFF, default ON
            --with_java=*)
                WITH_JAVA="${i#*=}"
                shift
                ;;
            # ON or OFF, default OFF
            --with_strip=*)
                WITH_STRIP="${i#*=}"
                shift
                ;;
            # string, absolute path to optimized model dir
            --opt_model_dir=*)
                OPTMODEL_DIR="${i#*=}"
                shift
                ;;
            # ON or OFF, default ON
            --shutdown_log=*)
                SHUTDOWN_LOG="${i#*=}"
                shift
                ;;
            # compiling lib which can operate on opencl and cpu.
            --with_opencl=*)
                WITH_OPENCL="${i#*=}"
                shift
                ;;
            # compiling lib which can operate on huawei npu.
            --with_huawei_kirin_npu=*)
                WITH_HUAWEI_KIRIN_NPU="${i#*=}"
                shift
                ;;
            --huawei_kirin_npu_sdk_root=*)
                HUAWEI_KIRIN_NPU_SDK_ROOT="${i#*=}"
                shift
                ;;
            # compiling result contains both light_api and cxx_api lib.
            full_publish)
                make_full_publish_so
                exit 0
                ;;
            # compiling lib with training ops.
            --with_train=*)
                WITH_TRAIN="${i#*=}"
                shift
                ;;
            help)
            # print help info
                print_usage
                exit 0
                ;;
            *)
                # unknown option
                echo "Error: unsupported argument \"${i#*=}\""
                print_usage
                exit 1
                ;;
        esac
        # compiling result contains light_api lib only, recommanded.
        make_tiny_publish_so
    done
}

main $@
