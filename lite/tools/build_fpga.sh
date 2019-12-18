#!/bin/bash

build_dir=build_fpga
mkdir -p ${build_dir}

root_dir=$(pwd)
build_dir=${build_dir}
# in build directory
# 1. Prepare gen_code file
GEN_CODE_PATH_PREFIX=${build_dir}/lite/gen_code
mkdir -p ${GEN_CODE_PATH_PREFIX}
touch ${GEN_CODE_PATH_PREFIX}/__generated_code__.cc

cd ${build_dir}
cmake .. \
        -DWITH_GPU=OFF \
        -DWITH_MKL=OFF \
        -DWITH_LITE=ON \
        -DLITE_WITH_CUDA=OFF \
        -DLITE_WITH_X86=OFF \
        -DLITE_WITH_ARM=ON \
        -DLITE_WITH_FPGA=ON \
        -DLITE_WITH_OPENMP=ON \
        -DLITE_WITH_LIGHT_WEIGHT_FRAMEWORK=ON \
        -DWITH_TESTING=OFF \
        -DARM_TARGET_OS=armlinux \
        -DLITE_BUILD_EXTRA=ON \
        -DLITE_WITH_PROFILE=OFF

make -j42
cd -
