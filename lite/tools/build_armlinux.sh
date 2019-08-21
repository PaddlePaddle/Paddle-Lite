#!/bin/bash

os=armlinux
abi=armv8
lang=gcc

if [ x$1 != x ]; then
  abi=$1
fi

if [ x$2 != x ]; then
  lang=$2
fi

cur_dir=$(pwd)
build_dir=$cur_dir/build.lite.${os}.${abi}.${lang}
mkdir -p $build_dir
cd $build_dir

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
        -DWITH_ARM_DOTPROD=ON   \
        -DLITE_WITH_OPENMP=ON \
        -DLITE_WITH_LIGHT_WEIGHT_FRAMEWORK=ON \
        -DWITH_TESTING=ON \
        -DARM_TARGET_OS=${os} -DARM_TARGET_ARCH_ABI=${abi} -DARM_TARGET_LANG=${lang}

make -j4 publish_inference

cd -
