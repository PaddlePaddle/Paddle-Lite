#! bin/bash
BIN_PATH="/data/local/tmp/lite/profile"
echo "-- Paddle run binary path: $BIN_PATH"
echo "model name: ${2}, input_shape: ${4}, out_txt: ${3}, threads: ${4}, pull_dir: $BIN_PATH/${3}"
echo "arm_abi: ${8}"
if [ ${8} == '1' ]; then
    echo "v8"
    adb -s ${1} shell "cd $BIN_PATH && ../tf_model/test_model_bin_v8 --model_dir=./tf_model/${2} --input_shape=${4} --out_txt=${3} --warmup=${5} --repeats=${6} --threads=${7} --use_optimize_nb=0 --power_mode=3 && exit" #
else 
    echo "v7"
    adb -s ${1} shell "cd $BIN_PATH && ../tf_model/test_model_bin_v7 --model_dir=./tf_model/${2} --input_shape=${4} --out_txt=${3} --warmup=${5} --repeats=${6} --threads=${7} --use_optimize_nb=0 --power_mode=3 && exit" #
fi 
