#!/usr/bin/env bash

operators=

function AddTest() {
  operators="${operators} $1"
}

function ExecuteAndroidTest() {
  platform=$1
  adb shell rm -rf /data/local/tmp/*
  adb push ../build/${platform}/build/libpaddle-mobile.so /data/local/tmp/
  for op in ${operators}
  do
    adb push ../test/build/test-${op}-op /data/local/tmp/
    adb shell "cd /data/local/tmp/; LD_LIBRARY_PATH=. ./test-${op}-op"
    echo "${BLUE}run test ${op} pass${NONE}"
  done
}

AddTest batchnorm
AddTest cast
AddTest conv
AddTest dequantize
#AddTest elementwiseadd
AddTest log
AddTest logical-and
AddTest logical-not
AddTest logical-or
AddTest logical-xor
AddTest pool
AddTest quantize
AddTest relu
AddTest relu6
AddTest sequence-expand
AddTest sequence-pool
AddTest sequence-softmax
AddTest sigmoid
AddTest softmax
AddTest tanh
AddTest topk
