#!/bin/bash

readonly CMAKE_COMMON_OPTIONS="-DWITH_GPU=OFF \
                               -DWITH_MKL=OFF \
                               -DWITH_LITE=ON \
                               -DLITE_WITH_CUDA=OFF \
                               -DLITE_WITH_X86=OFF \
                               -DLITE_WITH_ARM=ON \
                               -DLITE_WITH_LIGHT_WEIGHT_FRAMEWORK=ON"

# for code gen, a source file is generated after a test, but is dependended by some targets in cmake.
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

function make_tiny_publish_so {
  local os=$1
  local abi=$2
  local lang=$3
  local android_stl=$4

  cur_dir=$(pwd)
  build_dir=$cur_dir/build.lite.${os}.${abi}.${lang}
  mkdir -p $build_dir
  cd $build_dir

  cmake .. \
      ${CMAKE_COMMON_OPTIONS} \
      -DWITH_TESTING=OFF \
      -DLITE_WITH_JAVA=ON \
      -DLITE_SHUTDOWN_LOG=ON \
      -DLITE_ON_TINY_PUBLISH=ON \
      -DANDROID_STL_TYPE=$android_stl \
      -DARM_TARGET_OS=${os} -DARM_TARGET_ARCH_ABI=${abi} -DARM_TARGET_LANG=${lang}

  make publish_inference -j4
  cd - > /dev/null
}

function make_full_publish_so {
  local os=$1
  local abi=$2
  local lang=$3
  local android_stl=$4

  cur_dir=$(pwd)
  build_dir=$cur_dir/build.lite.${os}.${abi}.${lang}
  mkdir -p $build_dir
  cd $build_dir

  prepare_workspace
  cmake .. \
      ${CMAKE_COMMON_OPTIONS} \
      -DWITH_TESTING=OFF \
      -DLITE_WITH_JAVA=ON \
      -DLITE_SHUTDOWN_LOG=ON \
      -DANDROID_STL_TYPE=$android_stl \
      -DARM_TARGET_OS=${os} -DARM_TARGET_ARCH_ABI=${abi} -DARM_TARGET_LANG=${lang}

  make publish_inference -j4
  cd - > /dev/null
}

function make_all_tests {
  local os=$1
  local abi=$2
  local lang=$3

  cur_dir=$(pwd)
  build_dir=$cur_dir/build.lite.${os}.${abi}.${lang}
  mkdir -p $build_dir
  cd $build_dir

  prepare_workspace
  cmake .. \
      ${CMAKE_COMMON_OPTIONS} \
      -DWITH_TESTING=ON \
      -DARM_TARGET_OS=${os} -DARM_TARGET_ARCH_ABI=${abi} -DARM_TARGET_LANG=${lang}

  make lite_compile_deps -j4
  cd - > /dev/null
}


function print_usage {
    echo -e "\nUSAGE:"
    echo
    echo "----------------------------------------"
    echo -e "compile tiny publish so lib:"
    echo -e "   ./build.sh --arm_os=<os> --arm_abi=<abi> --arm_lang=<lang> --android_stl=<stl> tiny_publish"
    echo
    echo -e "compile full publish so lib:"
    echo -e "   ./build.sh --arm_os=<os> --arm_abi=<abi> --arm_lang=<lang> --android_stl=<stl> full_publish"
    echo
    echo -e "compile all arm tests:"
    echo -e "   ./build.sh --arm_os=<os> --arm_abi=<abi> --arm_lang=<lang> test"
    echo
    echo "----------------------------------------"
    echo
}

function main {
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
            --arm_lang=*)
                ARM_LANG="${i#*=}"
                shift
                ;;
            --android_stl=*)
                ANDROID_STL="${i#*=}"
                shift
                ;;
            tiny_publish)
                make_tiny_publish_so $ARM_OS $ARM_ABI $ARM_LANG $ANDROID_STL
                shift
                ;;
            full_publish)
                make_full_publish_so $ARM_OS $ARM_ABI $ARM_LANG $ANDROID_STL
                shift
                ;;
            test)
                make_all_tests $ARM_OS $ARM_ABI $ARM_LANG
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
