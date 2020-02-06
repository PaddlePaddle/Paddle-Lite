# #! bin/bash
# LITE_ROOT="/home/chenjiao04/vis_shoubai/Paddle-Lite"
LITE_ROOT="/home/chenjiao04/dev_my/Paddle-Lite"
echo "-- Paddle lite root dir is: $LITE_ROOT"

BIN_PATH="/data/local/tmp/lite/vis_shoubai"
echo "-- Paddle run binary path: $BIN_PATH"

MODLE_PATH="/home/chenjiao04/shoubai_cpu"
echo "-- model path: $MODLE_PATH"

# cd ..
# # echo "armv8 building:"
# # sh $LITE_ROOT/lite/tools/build.sh --arm_os=android --arm_abi=armv8 --arm_lang=gcc test
# # echo "armv7 building:"
# # sh $LITE_ROOT/lite/tools/build.sh --arm_os=android --arm_abi=armv7 --arm_lang=gcc test

# # echo "optimize model:"
# # sh $LITE_ROOT/lite/tools/build.sh build_optimize_tool
echo "optimizing model: "
echo "mnasnet"
$LITE_ROOT/build.model_optimize_tool/lite/api/model_optimize_tool --optimize_out_type=naive_buffer --optimize_out=$MODLE_PATH/mnasnet --model_dir=$MODLE_PATH/mnasnet
echo "Lens_MnasNet"
$LITE_ROOT/build.model_optimize_tool/lite/api/model_optimize_tool --optimize_out_type=naive_buffer --optimize_out=$MODLE_PATH/Lens_MnasNet --model_file=$MODLE_PATH/Lens_MnasNet/model --param_file=$MODLE_PATH/Lens_MnasNet/params
echo "Lens_YoloNano"
$LITE_ROOT/build.model_optimize_tool/lite/api/model_optimize_tool --optimize_out_type=naive_buffer --optimize_out=$MODLE_PATH/Lens_YoloNano --model_file=$MODLE_PATH/Lens_YoloNano/model --param_file=$MODLE_PATH/Lens_MnasNet/params
echo "picture_chaofen"
$LITE_ROOT/build.model_optimize_tool/lite/api/model_optimize_tool --optimize_out_type=naive_buffer --optimize_out=$MODLE_PATH/picture_chaofen --model_file=$MODLE_PATH/picture_chaofen/model --param_file=$MODLE_PATH/picture_chaofen/params
echo "saoma_v2"
$LITE_ROOT/build.model_optimize_tool/lite/api/model_optimize_tool --optimize_out_type=naive_buffer --optimize_out=$MODLE_PATH/saoma_v2 --model_file=$MODLE_PATH/saoma_v2/model --param_file=$MODLE_PATH/saoma_v2/params

echo "vis_shoubai model: "
# echo "ar_cup_detection_int8/detection"
# $LITE_ROOT/build.model_optimize_tool/lite/api/model_optimize_tool --optimize_out_type=naive_buffer --optimize_out=$MODLE_PATH/ar_cup_detection_int8/detection --model_file=$MODLE_PATH/ar_cup_detection_int8/detection/model --param_file=$MODLE_PATH/ar_cup_detection_int8/detection/params --prefer_int8_kernel=true
# echo "ar_cup_detection_int8/track"
# $LITE_ROOT/build.model_optimize_tool/lite/api/model_optimize_tool --optimize_out_type=naive_buffer --optimize_out=$MODLE_PATH/ar_cup_detection_int8/track --model_file=$MODLE_PATH/ar_cup_detection_int8/track/model --param_file=$MODLE_PATH/ar_cup_detection_int8/track/params --prefer_int8_kernel=true
# echo "automl_mv3_5ms_64_s_ftdongxiao_shoubai"
# $LITE_ROOT/build.model_optimize_tool/lite/api/model_optimize_tool --optimize_out_type=naive_buffer --optimize_out=$MODLE_PATH/automl_mv3_5ms_64_s_ftdongxiao_shoubai --model_file=$MODLE_PATH/automl_mv3_5ms_64_s_ftdongxiao_shoubai/model --param_file=$MODLE_PATH/automl_mv3_5ms_64_s_ftdongxiao_shoubai/params --prefer_int8_kernel=false
# echo "eye_mv1s_infer"
# $LITE_ROOT/build.model_optimize_tool/lite/api/model_optimize_tool --optimize_out_type=naive_buffer --optimize_out=$MODLE_PATH/eye_mv1s_infer --model_file=$MODLE_PATH/eye_mv1s_infer/model --param_file=$MODLE_PATH/eye_mv1s_infer/params --prefer_int8_kernel=false
# echo "handkeypoints/kpt_model_detection"
# $LITE_ROOT/build.model_optimize_tool/lite/api/model_optimize_tool --optimize_out_type=naive_buffer --optimize_out=$MODLE_PATH/handkeypoints/kpt_model_detection --model_file=$MODLE_PATH/handkeypoints/kpt_model_detection/model --param_file=$MODLE_PATH/handkeypoints/kpt_model_detection/params --prefer_int8_kernel=false
# echo "handkeypoints/kpt_model_keypoints"
# $LITE_ROOT/build.model_optimize_tool/lite/api/model_optimize_tool --optimize_out_type=naive_buffer --optimize_out=$MODLE_PATH/handkeypoints/kpt_model_keypoints --model_file=$MODLE_PATH/handkeypoints/kpt_model_keypoints/model --param_file=$MODLE_PATH/handkeypoints/kpt_model_keypoints/params --prefer_int8_kernel=true
# echo "models_0158"
# $LITE_ROOT/build.model_optimize_tool/lite/api/model_optimize_tool --optimize_out_type=naive_buffer --optimize_out=$MODLE_PATH/models_0158 --model_file=$MODLE_PATH/models_0158/model --param_file=$MODLE_PATH/models_0158/params --prefer_int8_kernel=false
# echo "mouth_mv6_epoch320_shoubai"
# $LITE_ROOT/build.model_optimize_tool/lite/api/model_optimize_tool --optimize_out_type=naive_buffer --optimize_out=$MODLE_PATH/mouth_mv6_epoch320_shoubai --model_file=$MODLE_PATH/mouth_mv6_epoch320_shoubai/model --param_file=$MODLE_PATH/mouth_mv6_epoch320_shoubai/params --prefer_int8_kernel=false
# echo "mv3_gp_shoubai"
# $LITE_ROOT/build.model_optimize_tool/lite/api/model_optimize_tool --optimize_out_type=naive_buffer --optimize_out=$MODLE_PATH/mv3_gp_shoubai --model_file=$MODLE_PATH/mv3_gp_shoubai/model --param_file=$MODLE_PATH/mv3_gp_shoubai/params --prefer_int8_kernel=false
# echo "mv8_angle_shoubai"
# $LITE_ROOT/build.model_optimize_tool/lite/api/model_optimize_tool --optimize_out_type=naive_buffer --optimize_out=$MODLE_PATH/mv8_angle_shoubai --model_file=$MODLE_PATH/mv8_angle_shoubai/model --param_file=$MODLE_PATH/mv8_angle_shoubai/params --prefer_int8_kernel=false
# echo "skyseg_shufflenet_0520_160"
# $LITE_ROOT/build.model_optimize_tool/lite/api/model_optimize_tool --optimize_out_type=naive_buffer --optimize_out=$MODLE_PATH/skyseg_shufflenet_0520_160 --model_dir=$MODLE_PATH/skyseg_shufflenet_0520_160 --prefer_int8_kernel=false
# echo "int8"
# $LITE_ROOT/build.model_optimize_tool/lite/api/model_optimize_tool --optimize_out_type=naive_buffer --optimize_out=$MODLE_PATH/int8 --model_file=$MODLE_PATH/int8/model --param_file=$MODLE_PATH/int8/weights --prefer_int8_kernel=true
# echo "merge21_ssd_shufflenet_quant-fluild"
# $LITE_ROOT/build.model_optimize_tool/lite/api/model_optimize_tool --optimize_out_type=naive_buffer --optimize_out=$MODLE_PATH/merge21_ssd_shufflenet_quant-fluild --model_file=$MODLE_PATH/merge21_ssd_shufflenet_quant-fluild/model --param_file=$MODLE_PATH/merge21_ssd_shufflenet_quant-fluild/params --prefer_int8_kernel=true
# echo "merge21-sbl-shufflenet-fluid"
# $LITE_ROOT/build.model_optimize_tool/lite/api/model_optimize_tool --optimize_out_type=naive_buffer --optimize_out=$MODLE_PATH/merge21-sbl-shufflenet-fluid --model_file=$MODLE_PATH/merge21-sbl-shufflenet-fluid/model --param_file=$MODLE_PATH/merge21-sbl-shufflenet-fluid/params --prefer_int8_kernel=true

# echo "paddle_pose_merged_models"
# mv1="/home/chenjiao04/vis_shoubai/paddle_pose_merged_models"
# mv2="/home/chenjiao04/vis_shoubai/paddle_gesture_merged_models"
# echo "paddle_pose_merged_models"
# $LITE_ROOT/build.model_optimize_tool/lite/api/model_optimize_tool --optimize_out_type=naive_buffer --optimize_out=$mv1/v1_7_person_det_merged_paddle --model_file=$mv1/v1_7_person_det_merged_paddle/__model__ --param_file=$mv1/v1_7_person_det_merged_paddle/__params__ 
# $LITE_ROOT/build.model_optimize_tool/lite/api/model_optimize_tool --optimize_out_type=naive_buffer --optimize_out=$mv1/v2_6_2dpose_dlc_merged_paddle --model_file=$mv1/v2_6_2dpose_dlc_merged_paddle/__model__ --param_file=$mv1/v2_6_2dpose_dlc_merged_paddle/__params__ 
# $LITE_ROOT/build.model_optimize_tool/lite/api/model_optimize_tool --optimize_out_type=naive_buffer --optimize_out=$mv1/v2_4_2dpose_anakin_merged_paddle --model_file=$mv1/v2_4_2dpose_anakin_merged_paddle/__model__ --param_file=$mv1/v2_4_2dpose_anakin_merged_paddle/__params__ 

# echo "paddle_gesture_merged_models"
# $LITE_ROOT/build.model_optimize_tool/lite/api/model_optimize_tool --optimize_out_type=naive_buffer --optimize_out=$mv2/v2_2_gesture_cls_merged_paddle --model_file=$mv2/v2_2_gesture_cls_merged_paddle/__model__ --param_file=$mv2/v2_2_gesture_cls_merged_paddle/__params__ 
# $LITE_ROOT/build.model_optimize_tool/lite/api/model_optimize_tool --optimize_out_type=naive_buffer --optimize_out=$mv2/v5_3_5_gesture_det_merged_paddle --model_file=$mv2/v5_3_5_gesture_det_merged_paddle/__model__ --param_file=$mv2/v5_3_5_gesture_det_merged_paddle/__params__ 
# $LITE_ROOT/build.model_optimize_tool/lite/api/model_optimize_tool --optimize_out_type=naive_buffer --optimize_out=$mv2/v1_fingertip_pose_merge_paddle --model_file=$mv2/v1_fingertip_pose_merge_paddle/__model__ --param_file=$mv2/v1_fingertip_pose_merge_paddle/__params__ 

# cd ./run_model_py