#!/bin/bash

# Script to download model files and copy .Jar and JNI lib for Android demo
# $1 will be the arch name

if [ $1 == "armv8" ]; then
  cp ../../../java/so/libpaddle_lite_jni.so PaddlePredictor/app/src/main/jniLibs/arm64-v8a/
elif [ $1 == "armv7" ]; then
  cp ../../../java/so/libpaddle_lite_jni.so PaddlePredictor/app/src/main/jniLibs/armeabi-v7a/
else
  echo "Error! arch type should be inputed. Arch type can be armv8 or armv7."
fi

MODELS=(inception_v4_simple_opt.nb lite_naive_model_opt.nb mobilenet_v1_opt.nb mobilenet_v2_relu_opt.nb resnet50_opt.nb)
MODELS_DIR=PaddlePredictor/app/src/main/assets/

for m in "${MODELS[@]}"
do
  wget --no-check-certificate -q http://paddlelite-demo.bj.bcebos.com/demo/java/${m} \
      -O ${MODELS_DIR}${m}
  rm -rf ${MODELS_DIR}${m}.tar.gz
done

cp ../../../java/jar/PaddlePredictor.jar PaddlePredictor/app/libs/
echo "Success: Asserts of this APP is already prepared!"
