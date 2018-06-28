#!/usr/bin/env sh

push_fn () {
MODELS_PATH="../../test/models/*"
MODELS_SRC="../../test/models"
IMAGE_PATH="../../test/images/*"
EXE_FILE="../../test/build/*"
EXE_DIR="data/local/tmp/bin"
adb shell mkdir ${EXE_DIR}
MODELS_DIR="data/local/tmp/models"
adb shell mkdir ${MODELS_DIR}
for file in `ls ${MODELS_SRC}`
do 
    adb shell mkdir ${MODELS_DIR}"/"${file}
done

IMAGES_DIR="data/local/tmp/images"
adb shell mkdir ${IMAGES_DIR}
LIB_PATH="../../build/release/arm-v7a/build/*"
adb push ${EXE_FILE} ${EXE_DIR}
adb push ${LIB_PATH} ${EXE_DIR}
if [[ $1 != "npm" ]]; then
adb push ${IMAGE_PATH} ${IMAGES_DIR}
adb push ${MODELS_PATH} ${MODELS_DIR}
fi
echo "test-op or test-net below : "
adb shell ls /data/local/tmp/bin
echo "**** choose OP or NET to test ****"
read -p "which to test : " test_name
adb shell "cd /data/local/tmp/bin; LD_LIBRARY_PATH=. ./${test_name}"
}

if [[ $1 == "npm" ]]; then
push_fn $1
else
push_fn
fi