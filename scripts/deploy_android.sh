#!/usr/bin/env sh

test_fn () {
    EXE_FILE="mdlTest"
    DATA_PATH="./model/"

    EXE_DIR="/data/local/tmp"
    adb push ${EXE_FILE} ${EXE_DIR}
    adb push ${DATA_PATH} ${EXE_DIR}
    echo "test files sync completed"
}
test_fn
