#!/bin/bash
set -e

build_dir=build.ios.armv7.arm64
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
        -DARM_TARGET_OS=ios

make -j4

cd -
