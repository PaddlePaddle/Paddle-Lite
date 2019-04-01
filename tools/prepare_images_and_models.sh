#!/usr/bin/env bash

# decalre download paths of images and models
PADDLE_MOBILE_ROOT="$(pwd)/../"
IMAGES_AND_MODELS="opencl_test_src"
IMAGES_AND_MODELS_PATH="http://mms-graph.bj.bcebos.com/paddle-mobile/${IMAGES_AND_MODELS} ${PADDLE_MOBILE_ROOT}/download/${IMAGES_AND_MODELS}.zip"

# download and unzip zip-files of images and models
mkdir ${PADDLE_MOBILE_ROOT}/download/
cd ${PADDLE_MOBILE_ROOT}/download/
wget -c ${IMAGES_AND_MODELS_PATH}
unzip -o ./${IMAGES_AND_MODELS}.zip

# create models and images directories below test
mkdir ${PADDLE_MOBILE_ROOT}/test/models
mkdir ${PADDLE_MOBILE_ROOT}/test/images

# move to test directory
cp ./${IMAGES_AND_MODELS}/input_3x224x224_banana ${PADDLE_MOBILE_ROOT}/test/images/
cp -r ./${IMAGES_AND_MODELS}/mobilenet ${PADDLE_MOBILE_ROOT}/test/models/
