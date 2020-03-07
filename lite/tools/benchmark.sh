#!/bin/bash
set -e

# Check input
if [ $# -lt  2 ];
then
    echo "Input error"
    echo "Usage:"
    echo "  sh benchmark.sh benchmark_bin_path benchmark_models_path <result_filename> <input_shape> <power_mode: [0|1|2|3]> <is_run_model_optimize: [true|false]> <is_run_quantized_model: [trur|false]>"
    echo "\npower_mode refer: 0 for big cluster, 1 for little cluster, 2 for all cores,  3 for no bind."
    exit
fi

# Set benchmark params
ANDROID_DIR=/data/local/tmp
BENCHMARK_BIN=$1
MODELS_DIR=$2

RESULT_FILENAME=result.txt
INPUT_SHAPE=1,3,244,244
POWER_MODE=3
WARMUP=10
REPEATS=30
IS_RUN_MODEL_OPTIMIZE=false
IS_RUN_QUANTIZED_MODEL=false
NUM_THREADS_LIST=(1 2 4)
MODELS_LIST=$(ls $MODELS_DIR)

# Check input
if [ $# -gt  2 ];
then
    RESULT_FILENAME=$3
fi
if [ $# -gt  3 ];
then
    INPUT_SHAPE=$4
fi
if [ $# -gt  4 ];
then
    POWER_MODE=$5
fi
if [ $# -gt  5 ];
then
    IS_RUN_MODEL_OPTIMIZE=$6
fi
if [ $# -gt  6 ];
then
    IS_RUN_QUANTIZED_MODEL=$7
fi

# Adb push benchmark_bin, models
adb push $BENCHMARK_BIN $ANDROID_DIR/benchmark_bin
adb shell chmod +x $ANDROID_DIR/benchmark_bin
adb push $MODELS_DIR $ANDROID_DIR

# Run benchmark
adb shell "echo 'PaddleLite Benchmark (in ms)\n' > $ANDROID_DIR/$RESULT_FILENAME"
for threads in ${NUM_THREADS_LIST[@]}; do
    adb shell "echo threads=$threads warmup=$WARMUP repeats=$REPEATS input_shape=$INPUT_SHAPE power_mode=$POWER_MODE >> $ANDROID_DIR/$RESULT_FILENAME"
    for model_name in ${MODELS_LIST[@]}; do
      echo "Model=$model_name Threads=$threads"
      adb shell "$ANDROID_DIR/benchmark_bin \
                   --model_dir=$ANDROID_DIR/${MODELS_DIR}/$model_name \
                   --input_shape=$INPUT_SHAPE \
                   --warmup=$WARMUP \
                   --repeats=$REPEATS \
                   --threads=$threads \
                   --power_mode=$POWER_MODE \
                   --result_filename=$ANDROID_DIR/$RESULT_FILENAME \
                   --run_model_optimize=$IS_RUN_MODEL_OPTIMIZE \
                   --is_quantized_model=$IS_RUN_QUANTIZED_MODEL"
    done
    adb shell "echo >> $ANDROID_DIR/$RESULT_FILENAME"
done
adb shell "echo >> $ANDROID_DIR/$RESULT_FILENAME"
adb shell "echo power_mode refer: 0 for big cluster, 1 for little cluster, 2 for all cores,  3 for no bind >> $ANDROID_DIR/$RESULT_FILENAME"
# Adb pull benchmark result, show result
adb pull $ANDROID_DIR/$RESULT_FILENAME .
echo "\n--------------------------------------"
cat $RESULT_FILENAME
echo "--------------------------------------"
