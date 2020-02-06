#! bin/bash
MODLE_PATH="/home/chenjiao04/tf_model"
BIN_PATH="/data/local/tmp/lite/profile"
echo "-- model path: $MODLE_PATH"
echo "-- Paddle run binary path: $BIN_PATH"

#adb -s ${1} shell "cd /data/local/tmp/lite && mkdir profile && exit"
#adb -s ${1} shell "cd /data/local/tmp/lite/profile && mkdir tf_model && exit"
#echo "adb push model"
#adb -s ${1} push  $MODLE_PATH $BIN_PATH

#echo "adb push model_test"
#UNIT_V7_TEST_ROOT="/home/chenjiao04/Paddle-Lite/Paddle-Lite/build.lite.android.armv7.gcc_1"
#UNIT_V8_TEST_ROOT="/home/chenjiao04/Paddle-Lite/Paddle-Lite/build.lite.android.armv8.gcc_1"

#echo "v7 = $UNIT_V7_TEST_ROOT"
#adb -s ${1} push $UNIT_V7_TEST_ROOT/lite/api/test_model_bin $BIN_PATH
#adb -s ${1} shell "cd $BIN_PATH && mv test_model_bin test_model_bin_v7 && rm *.txt && exit"

#echo "adb push mnn model test"
#UNIT_V7_TEST_ROOT="/home/chenjiao04/MNN/project/android/build_32"
#UNIT_V8_TEST_ROOT="/home/chenjiao04/MNN/project/android/build_64"
#adb -s ${1} shell "cd $BIN_PATH && mkdir mnn_v7 && exit"
#adb -s ${1} shell "cd $BIN_PATH && mkdir mnn_v8 && exit"
#echo "v7 = $UNIT_V7_TEST_ROOT"
#adb -s ${1} push $UNIT_V7_TEST_ROOT/MNNV2Basic.out $BIN_PATH/mnn_v7
#adb -s ${1} push $UNIT_V7_TEST_ROOT/libMNN.so $BIN_PATH/mnn_v7
#adb -s ${1} push $UNIT_V7_TEST_ROOT/libMNN_Express.so $BIN_PATH/mnn_v7

#echo "v8 = $UNIT_V8_TEST_ROOT"
#adb -s ${1} push $UNIT_V8_TEST_ROOT/MNNV2Basic.out $BIN_PATH/mnn_v8
#adb -s ${1} push $UNIT_V8_TEST_ROOT/libMNN.so $BIN_PATH/mnn_v8
#adb -s ${1} push $UNIT_V8_TEST_ROOT/libMNN_CL.so $BIN_PATH/mnn_v8
#adb -s ${1} push $UNIT_V8_TEST_ROOT/libMNN_Express.so $BIN_PATH/mnn_v8
