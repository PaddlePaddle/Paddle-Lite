#!/usr/bin/env sh

push_fn () {
MODELS_PATH="../../test/models/*"
MODELS_SRC="../../test/models"
IMAGE_PATH="../../test/images/*"
EXE_FILE="../../test/build/*"
EXE_DIR="/data/local/tmp/bin"
adb shell mkdir ${EXE_DIR}
MODELS_DIR="/data/local/tmp/models"
adb shell mkdir ${MODELS_DIR}
for file in `ls ${MODELS_SRC}`
do
    adb shell mkdir ${MODELS_DIR}"/"${file}
done

if [[ -d "../../src/operators/kernel/mali/ACL_Android/build" ]]; then
ACL_BUILD_PATH="../../src/operators/kernel/mali/ACL_Android/build/*"
adb push ${ACL_BUILD_PATH} ${EXE_DIR}
fi

IMAGES_DIR="/data/local/tmp/images"
adb shell mkdir ${IMAGES_DIR}
LIB_PATH="../../build/release/arm-v7a/build/*"
#LIB_PATH="../../build/release/arm-v8a/build/*"
adb push ${EXE_FILE} ${EXE_DIR}
for file in ${LIB_PATH}
do
    adb push ${file} ${EXE_DIR}
done

if [[ $1 != "npm" ]]; then
adb push ${IMAGE_PATH} ${IMAGES_DIR}
adb push ${MODELS_PATH} ${MODELS_DIR}
fi
}

if [[ $1 == "npm" ]]; then
push_fn $1
else
push_fn
fi
