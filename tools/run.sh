#!/usr/bin/env sh
# auto build and run

BUILDNET="mobilenetssd"
TESTUNIT="test-mobilenetssd"

push_fn () {
sh build.sh android ${BUILDNET}
MODELS_PATH="../test/models/*"
MODELS_SRC="../test/models"
IMAGE_PATH="../test/images/*"
EXE_FILE="../test/build/*"
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
LIB_PATH="../build/release/arm-v7a/build/*"
adb push ${EXE_FILE} ${EXE_DIR}
adb push ${LIB_PATH} ${EXE_DIR}
if [[ $1 != "pm" ]]; then
adb push ${IMAGE_PATH} ${IMAGES_DIR}
adb push ${MODELS_PATH} ${MODELS_DIR}
fi
adb shell "cd /data/local/tmp/bin; LD_LIBRARY_PATH=. ./${TESTUNIT}"
}

if [[ $1 == "pm" ]]; then
push_fn $1
else
push_fn
fi
