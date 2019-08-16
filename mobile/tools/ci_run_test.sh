#!/usr/bin/env bash

operators=

function AddTest() {
  operators="${operators} $1"
}

function ExecuteAndroidTests() {
  platform=$1
  devices=`adb devices | grep -v devices | grep device | awk -F ' ' '{print $1}'`
  for device in ${devices}; do
    adb -s ${device} shell rm -rf /data/local/tmp/*
    adb -s ${device} push ../build/${platform}/build/libpaddle-mobile.so /data/local/tmp/
    for op in ${operators}; do
      adb -s ${device} push ../test/build/test-${op}-op /data/local/tmp/
      adb -s ${device} shell "cd /data/local/tmp/; LD_LIBRARY_PATH=. ./test-${op}-op"
      echo "${BLUE}run test ${op} pass${NONE}"
    done
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
