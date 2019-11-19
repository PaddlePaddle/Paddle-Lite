#! bin/bash
MODLE_PATH="/home/chenjiao04/vis_shoubai"
echo "-- model path: $MODLE_PATH"
BIN_PATH="/data/local/tmp/lite/vis_shoubai"
echo "-- Paddle run binary path: $BIN_PATH"
# adb -s ${1} shell "cd /data/local/tmp/lite && mkdir vis_shoubai && cd vis_shoubai && mkdir model && exit"

echo "adb push model"
# adb -s ${1} push  $MODLE_PATH/ar_cup_detection_int8 $BIN_PATH/model 
# adb -s ${1} push  $MODLE_PATH/automl_mv3_5ms_64_s_ftdongxiao_shoubai $BIN_PATH/model
# adb -s ${1} push  $MODLE_PATH/eye_mv1s_infer $BIN_PATH/model
# adb -s ${1} push  $MODLE_PATH/handkeypoints $BIN_PATH/model
# adb -s ${1} push  $MODLE_PATH/int8 $BIN_PATH/model
# adb -s ${1} push  $MODLE_PATH/models_0158 $BIN_PATH/model
# adb -s ${1} push  $MODLE_PATH/mouth_mv6_epoch320_shoubai $BIN_PATH/model
# adb -s ${1} push  $MODLE_PATH/mv3_gp_shoubai $BIN_PATH/model
# adb -s ${1} push  $MODLE_PATH/mv8_angle_shoubai $BIN_PATH/model
# adb -s ${1} push  $MODLE_PATH/skyseg_shufflenet_0520_160 $BIN_PATH/model

echo "adb push model_test"
UNIT_V7_TEST_ROOT="/home/chenjiao04/vis_shoubai/Paddle-Lite/build.lite.android.armv7.gcc"
UNIT_V8_TEST_ROOT="/home/chenjiao04/vis_shoubai/Paddle-Lite/build.lite.android.armv8.gcc"
echo "v7 = $UNIT_V7_TEST_ROOT"
adb -s ${1} push $UNIT_V7_TEST_ROOT/lite/api/test_model_bin $BIN_PATH
adb -s ${1} shell "cd $BIN_PATH && mv test_model_bin test_model_bin_v7 && rm *.txt && exit"

echo "v8 = $UNIT_V8_TEST_ROOT"
adb -s ${1} push $UNIT_V8_TEST_ROOT/lite/api/test_model_bin $BIN_PATH
adb -s ${1} shell "cd $BIN_PATH && mv test_model_bin test_model_bin_v8 && exit"



