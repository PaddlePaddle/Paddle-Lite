#!/bin/bash
set -e

# Check input
if [ $# -lt  3 ];
then
    echo "Input error"
    echo "Usage:"
    echo "  sh benchmark.sh <benchmark_bin_path> <benchmark_models_path> <result_filename>"
    echo "  sh benchmark.sh <benchmark_bin_path> <benchmark_models_path> <result_filename> <is_run_model_optimize: [true|false]>"
    exit
fi

# Set benchmark params
ANDROID_DIR=/data/local/tmp
BENCHMARK_BIN=$1
MODELS_DIR=$2
RESULT_FILENAME=$3

WARMUP=10
REPEATS=30
IS_RUN_MODEL_OPTIMIZE=false
IS_RUN_QUANTIZED_MODEL=false
NUM_THREADS_LIST=(1 2 4)
MODELS_LIST=$(ls $MODELS_DIR)

# Check input
if [ $# -gt  3 ];
then
    IS_RUN_MODEL_OPTIMIZE=$4
fi

# Adb push benchmark_bin, models
adb push $BENCHMARK_BIN $ANDROID_DIR/benchmark_bin
adb shell chmod +x $ANDROID_DIR/benchmark_bin
adb push $MODELS_DIR $ANDROID_DIR

# Run benchmark
adb shell "echo 'PaddleLite Benchmark' > $ANDROID_DIR/$RESULT_FILENAME"
for threads in ${NUM_THREADS_LIST[@]}; do
    adb shell "echo Threads=$threads Warmup=$WARMUP Repeats=$REPEATS >> $ANDROID_DIR/$RESULT_FILENAME"
    for model_name in ${MODELS_LIST[@]}; do
      echo "Model=$model_name Threads=$threads"
      if [ "$IS_RUN_MODEL_OPTIMIZE" = true ]; 
      then
          adb shell "$ANDROID_DIR/benchmark_bin \
                   --model_dir=$ANDROID_DIR/${MODELS_DIR}/$model_name \
                   --warmup=$WARMUP \
                   --repeats=$REPEATS \
                   --threads=$threads \
                   --result_filename=$ANDROID_DIR/$RESULT_FILENAME"
      else
          adb shell "$ANDROID_DIR/benchmark_bin \
                   --optimized_model_path=$ANDROID_DIR/${MODELS_DIR}/$model_name \
                   --warmup=$WARMUP \
                   --repeats=$REPEATS \
                   --threads=$threads \
                   --result_filename=$ANDROID_DIR/$RESULT_FILENAME"
      fi
    done
    adb shell "echo >> $ANDROID_DIR/$RESULT_FILENAME"
done

# Adb pull benchmark result, show result
adb pull $ANDROID_DIR/$RESULT_FILENAME .
echo "\n--------------------------------------"
cat $RESULT_FILENAME
echo "--------------------------------------"
