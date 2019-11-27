#! bin/bash
BIN_PATH="/data/local/tmp/lite/vis_shoubai"
echo "-- Paddle run binary path: $BIN_PATH"
RESULT_ROOT="/home/chenjiao04/vis_shoubai/Paddle-Lite/run_model_py"
echo "-- Paddle run binary path: $BIN_PATH"
echo "model name: ${2}, input_shape: ${4}, out_txt: ${3}, threads: ${4}, pull_dir: $BIN_PATH/${3}"
echo "arm_abi: ${8}"
if [${8} == 0]; then
    adb -s ${1} shell "cd $BIN_PATH && ./test_model_bin_v7 --model_dir=./model/${2} --input_shape=${4} --out_txt=${3} --warmup=${5} --repeats=${6} --threads=${7} --use_optimize_nb=true --power_mode=0 && exit" #
else
    adb -s ${1} shell "cd $BIN_PATH && ./test_model_bin_v8 --model_dir=./model/${2} --input_shape=${4} --out_txt=${3} --warmup=${5} --repeats=${6} --threads=${7} --use_optimize_nb=true --power_mode=0 && exit" #
fi 

# adb -s ${1} pull $BIN_PATH/${3} $RESULT_ROOT

