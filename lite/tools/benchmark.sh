#!/bin/bash
set -e

if [ $# -lt  2 ];
then
    echo "Input error"
    echo "USAGE:"
    echo "  sh benchmark.sh benchmark_bin_path test_models_dir"
    echo "  sh benchmark.sh benchmark_bin_path test_models_dir arm_bi"
    exit
fi

BENCHMARK_BIN=$1
MODELS_DIR=$2
ARM_BI=$3
ANDROID_DIR=/data/local/tmp
RESULT_FILENAME="result.txt"
WARMUP=10
REPEATS=30

adb push $BENCHMARK_BIN $ANDROID_DIR/benchmark_bin
adb shell chmod 777 $ANDROID_DIR/benchmark_bin
adb push $MODELS_DIR $ANDROID_DIR

adb shell "echo  PaddleLite Benchmark > $ANDROID_DIR/$RESULT_FILENAME"
for threads in 1 2 4
do
adb shell "echo ABI=$ARM_BI Threads=$threads Warmup=$WARMUP Repeats=$REPEATS  >>  $ANDROID_DIR/$RESULT_FILENAME"
for model_name in `ls $MODELS_DIR`
do
  echo $model_name
  adb shell "$ANDROID_DIR/benchmark_bin --model_dir=$ANDROID_DIR/${MODELS_DIR##*/}/$model_name --warmup=$WARMUP --repeats=$REPEATS --threads=$threads --result_filename=$ANDROID_DIR/$RESULT_FILENAME"
done
adb shell "echo  >>  $ANDROID_DIR/$RESULT_FILENAME"
done
adb pull $ANDROID_DIR/$RESULT_FILENAME .
