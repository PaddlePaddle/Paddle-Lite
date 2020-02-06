#! bin/bash
BIN_PATH="/data/local/tmp/lite/profile"
echo "-- Paddle run binary path: $BIN_PATH"
echo "model name: ${2}, repeats: ${3}, threads: ${4}"
echo "arm_abi: ${5}"
if [ ${5} == '1' ]; then
    echo "v8"
    adb -s ${1} shell "cd $BIN_PATH && cd mnn_v8 && export LD_LIBRARY_PATH=./ && ./MNNV2Basic.out ../tf_model/${2} ${3} 0 0 ${4}  && exit" #
else 
    echo "v7"
    adb -s ${1} shell "cd $BIN_PATH && cd mnn_v7 && export LD_LIBRARY_PATH=./ && ./MNNV2Basic.out ../tf_model/${2} ${3} 0 0 ${4}  && exit" #
fi
