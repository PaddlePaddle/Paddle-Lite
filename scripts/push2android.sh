#!/usr/bin/env sh

push_fn () {
MODELS_PATH="../test/models/*"
EXE_FILE="../test/build/*"
EXE_DIR="data/local/tmp/bin"
MODELS_DIR="data/local/tmp/models"
LIB_PATH="../build/release/arm-v7a/build/*"
adb push ${EXE_FILE} ${EXE_DIR}
adb push ${LIB_PATH} ${EXE_DIR}
adb push ${MODELS_PATH} ${MODELS_DIR}
echo "test files sync completed"
}
push_fn
