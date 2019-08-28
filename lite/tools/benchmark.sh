#!/bin/bash
set -e

if [ $# -lt  3 ];
then
    echo "Input error"
    echo "USAGE:"
    echo "  sh benchmark.sh benchmark_bin_path benchmark_models_path result_filename"
    echo "  sh benchmark.sh benchmark_bin_path benchmark_models_path result_filename is_run_model_optimize"
    exit
fi

ANDROID_DIR=/data/local/tmp
WARMUP=10
REPEATS=30
BENCHMARK_BIN=$1
MODELS_DIR=$2
RESULT_FILENAME=$3
IS_RUN_MODEL_OPTIMIZE=false
if [ $# -gt  3 ];
then
    IS_RUN_MODEL_OPTIMIZE=$4
fi

adb push $BENCHMARK_BIN $ANDROID_DIR/benchmark_bin
adb shell chmod 777 $ANDROID_DIR/benchmark_bin
adb push $MODELS_DIR $ANDROID_DIR

adb shell "echo  PaddleLite Benchmark > $ANDROID_DIR/$RESULT_FILENAME"
for threads in 1 2 4
do
adb shell "echo Threads=$threads Warmup=$WARMUP Repeats=$REPEATS  >>  $ANDROID_DIR/$RESULT_FILENAME"
for model_name in `ls $MODELS_DIR`
do
  echo "Model=$model_name Threads=$threads"
  adb shell "$ANDROID_DIR/benchmark_bin --model_dir=$ANDROID_DIR/${MODELS_DIR##*/}/$model_name --warmup=$WARMUP --repeats=$REPEATS --threads=$threads --result_filename=$ANDROID_DIR/$RESULT_FILENAME --run_model_optimize=$IS_RUN_MODEL_OPTIMIZE"
done
adb shell "echo  >>  $ANDROID_DIR/$RESULT_FILENAME"
done
adb pull $ANDROID_DIR/$RESULT_FILENAME .
echo "\n--------------------------------------"
cat $RESULT_FILENAME
echo "--------------------------------------"
