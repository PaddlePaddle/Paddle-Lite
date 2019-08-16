#!/bin/bash

# Script to download model files and copy .Jar and JNI lib for Android demo
# $1 will be the arch name

if [ x$1 != x ]; then
  cp ../../../java/so/libpaddle_lite_jni.so PaddlePredictor/app/src/main/jniLibs/$1/
else
  echo "Warning: didn't copy JNI .so lib because arch name is empty"
fi

MODELS=(inception_v4_simple_opt.nb lite_naive_model_opt.nb mobilenet_v1_opt.nb mobilenet_v2_relu_opt.nb resnet50_opt.nb)
MODELS_DIR=PaddlePredictor/app/src/main/assets/

for m in "${MODELS[@]}"
do
  wget --no-check-certificate -q http://paddle-inference-dist.bj.bcebos.com/${m}.tar.gz \
      -O ${MODELS_DIR}${m}.tar.gz
  tar xzf ${MODELS_DIR}${m}.tar.gz -C ${MODELS_DIR}
  rm -rf ${MODELS_DIR}${m}.tar.gz
done

cp ../../../java/jar/PaddlePredictor.jar PaddlePredictor/app/libs/
