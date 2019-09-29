#!/bin/bash
set -ex

readonly CMAKE_COMMON_OPTIONS="-DWITH_GPU=OFF \
                               -DWITH_MKL=OFF \
                               -DWITH_LITE=ON \
                               -DLITE_WITH_CUDA=OFF \
                               -DLITE_WITH_X86=OFF \
                               -DLITE_WITH_ARM=ON \
                               -DLITE_WITH_LIGHT_WEIGHT_FRAMEWORK=ON"

readonly NUM_PROC=${LITE_BUILD_THREADS:-4}


# global variables
BUILD_EXTRA=OFF
BUILD_JAVA=ON
BUILD_DIR=$(pwd)

readonly THIRDPARTY_TAR=https://paddle-inference-dist.bj.bcebos.com/PaddleLite/third-party-05b862.tar.gz

readonly workspace=$PWD

# for code gen, a source file is generated after a test, but is dependended by some targets in cmake.
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

function prepare_thirdparty {
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

function build_model_optimize_tool {
    cd $workspace
    prepare_thirdparty
    mkdir -p build.model_optimize_tool
    cd build.model_optimize_tool
    cmake .. -DWITH_LITE=ON \
      -DLITE_ON_MODEL_OPTIMIZE_TOOL=ON \
      -DWITH_TESTING=OFF \
      -DLITE_BUILD_EXTRA=ON \
      -DWITH_MKL=OFF
    make model_optimize_tool -j$NUM_PROC
}

function make_tiny_publish_so {
  local os=$1
  local abi=$2
  local lang=$3
  local android_stl=$4

  cur_dir=$(pwd)
  build_directory=$cur_dir/build.lite.${os}.${abi}.${lang}
  if [ -d $build_directory ]
  then
    rm -rf $build_directory
  fi
  mkdir -p $build_directory
  cd $build_directory

  if [ ${os} == "armlinux" ]; then
    BUILD_JAVA=OFF
  fi

  cmake .. \
      ${CMAKE_COMMON_OPTIONS} \
      -DWITH_TESTING=OFF \
      -DLITE_WITH_JAVA=$BUILD_JAVA \
      -DLITE_SHUTDOWN_LOG=ON \
      -DLITE_ON_TINY_PUBLISH=ON \
      -DANDROID_STL_TYPE=$android_stl \
      -DLITE_BUILD_EXTRA=$BUILD_EXTRA \
      -DARM_TARGET_OS=${os} -DARM_TARGET_ARCH_ABI=${abi} -DARM_TARGET_LANG=${lang}

#  make publish_inference -j$NUM_PROC
  make publish_inference -j

  infer_dir=$build_directory/inference_lite_lib.${os}.${abi}
#  infer_tag=./inference_lite_lib.${os}.${abi}.extra_$BUILD_EXTRA.${lang}.$android_stl.tiny_publish
  if [ ${BUILD_EXTRA} == "ON"]; then
     infer_tag=./inference_lite_lib.${os}.${abi}.with_extra.${lang}.$android_stl.tiny_publish
  else
     infer_tag=./inference_lite_lib.${os}.${abi}.${lang}.$android_stl.tiny_publish
  fi

  publish_dir=$cur_dir/publish_inference
  mv $infer_dir $infer_tag
  tar zcf $infer_tag.tar.gz $infer_tag

  if [ ! -d $publish_dir ]
  then
     mkdir -p $publish_dir
  fi

  mv $infer_tag.tar.gz $publish_dir/
  cd ..
  rm -rf $build_dir

#  cd - > /dev/null
}

function make_full_publish_so {
  local os=$1
  local abi=$2
  local lang=$3
  local android_stl=$4

  #git submodule update --init --recursive
  prepare_thirdparty

  root_dir=$(pwd)
  build_directory=$BUILD_DIR/build.lite.${os}.${abi}.${lang}

  if [ -d $build_directory ]
  then
    rm -rf $build_directory
  fi
  mkdir -p $build_directory
  cd $build_directory
  
  if [ ${os} == "armlinux" ]; then
    BUILD_JAVA=OFF
  fi

  prepare_workspace $root_dir $build_directory
  cmake $root_dir \
      ${CMAKE_COMMON_OPTIONS} \
      -DWITH_TESTING=OFF \
      -DLITE_WITH_JAVA=$BUILD_JAVA \
      -DLITE_SHUTDOWN_LOG=ON \
      -DANDROID_STL_TYPE=$android_stl \
      -DLITE_BUILD_EXTRA=$BUILD_EXTRA \
      -DARM_TARGET_OS=${os} -DARM_TARGET_ARCH_ABI=${abi} -DARM_TARGET_LANG=${lang}

#  make publish_inference -j4
  make publish_inference -j

  infer_dir=$build_directory/inference_lite_lib.${os}.${abi}
#  infer_tag=./inference_lite_lib.${os}.${abi}.extra_$BUILD_EXTRA.${lang}.$android_stl.full_publish
  if [ ${BUILD_EXTRA} == "ON" ]; then
     infer_tag=./inference_lite_lib.${os}.${abi}.with_extra.${lang}.$android_stl.full_publish
  else
     infer_tag=./inference_lite_lib.${os}.${abi}.${lang}.$android_stl.full_publish
  fi



  if [ ! -d $root_dir/publish_inference ]
  then
  mkdir -p $root_dir/publish_inference
  fi
  publish_dir=$root_dir/publish_inference

  cd $build_directory
  mv $infer_dir $infer_tag
  tar zcf $infer_tag.tar.gz $infer_tag

  if [ ! -d $publish_dir ]
  then
  mkdir -p $publish_dir
  fi

  mv $infer_tag.tar.gz $publish_dir/
  cd ..
  rm -rf $build_directory
#  cd - > /dev/null
  
}

function make_FPGA_publish_so {
  prepare_thirdparty
  root_dir=$(pwd)
  build_directory=$root_dir/build.lite.armlinux.FPGA

  if [ -d $build_directory ]
  then
    rm -rf $build_directory
  fi
  mkdir -p $build_directory
  cd $build_directory

  prepare_workspace $root_dir $build_directory
  cmake .. \
        -DWITH_GPU=OFF \
        -DWITH_MKL=OFF \
        -DWITH_LITE=ON \
        -DLITE_WITH_CUDA=OFF \
        -DLITE_WITH_X86=OFF \
        -DLITE_WITH_ARM=ON \
        -DLITE_WITH_OPENMP=ON   \
        -DLITE_WITH_LIGHT_WEIGHT_FRAMEWORK=ON \
        -DWITH_TESTING=OFF \
        -DLITE_WITH_FPGA=ON \
        -DLITE_BUILD_EXTRA=$BUILD_EXTRA \
        -DARM_TARGET_OS=armlinux 
    make publish_inference -j

  infer_dir=$build_directory/inference_lite_lib.armlinux.armv8
#  infer_tag=./inference_lite_lib.FPGA.extra_$BUILD_EXTRA
  if [ ${BUILD_EXTRA} == "ON" ]; then
     infer_tag=./inference_lite_lib.armlinux.FPGA.with_extra
  else
     infer_tag=./inference_lite_lib.armlinux.FPGA
  fi

  if [ ! -d $root_dir/publish_inference ]
  then
  mkdir -p $root_dir/publish_inference
  fi
  publish_dir=$root_dir/publish_inference

  cd $build_directory
  mv $infer_dir $infer_tag
  tar zcf $infer_tag.tar.gz $infer_tag

  mv $infer_tag.tar.gz $publish_dir/
  cd ..
  rm -rf $build_directory

#    rm -rf $publish_dir
}

function main {

  root_dir=$(pwd)
  if [ -d $root_dir/publish_inference ]
  then
  rm -rf $root_dir/publish_inference
  fi


  for build_extra in "OFF" "ON"
  do
      BUILD_EXTRA=$build_extra
      make_FPGA_publish_so
  done



  for os in "android"
  do
     ARM_OS=$os
     for arm_abi in  "armv7" "armv8"
     do
	 ARM_ABI=$arm_abi
	 for arm_lang in "gcc"
         do
             ARM_LANG=$arm_lang
             for android_stl in "c++_static" "c++_shared"
             do
                   ANDROID_STL=$android_stl
	           for build_extra in "OFF" "ON"
                   do
                       BUILD_EXTRA=$build_extra
                       make_tiny_publish_so $ARM_OS $ARM_ABI $ARM_LANG $ANDROID_STL
                       make_full_publish_so $ARM_OS $ARM_ABI $ARM_LANG $ANDROID_STL
                   done
             done
         done
     done
  done
}

main $@
