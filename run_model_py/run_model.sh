#! bin/bash
# BIN_PATH="/data/local/tmp/lite/vis_shoubai"
BIN_PATH="/data/local/tmp/lite/tf_model"
echo "-- Paddle run binary path: $BIN_PATH"
RESULT_ROOT="/home/chenjiao04/vis_shoubai/Paddle-Lite/run_model_py"
echo "-- Paddle run binary path: $RESULT_ROOT"
echo "model name: ${2}, input_shape: ${4}, out_txt: ${3}, threads: ${4}, pull_dir: $BIN_PATH/${3}"
echo "arm_abi: ${8}"
if [ ${8} == '1' ]; then
    echo "v8"
    adb -s ${1} shell "cd $BIN_PATH && ./test_model_bin_v8 --model_dir=./shoubai_cpu/${2} --input_shape=${4} --out_txt=${3} --warmup=${5} --repeats=${6} --threads=${7} --use_optimize_nb=true --power_mode=3 && exit" #
    # if [ ${1} == '17c3cc34' ]; then
    #     adb -s ${1} shell "cd $BIN_PATH && ./test_model_bin_v8 --model_dir=./model/${2} --input_shape=${4} --out_txt=${3} --warmup=${5} --repeats=${6} --threads=${7} --use_optimize_nb=true --power_mode=3 && exit" #
    # else
    #     adb -s ${1} shell "cd $BIN_PATH && ./test_model_bin_v8 --model_dir=./model/${2} --input_shape=${4} --out_txt=${3} --warmup=${5} --repeats=${6} --threads=${7} --use_optimize_nb=true && exit" #--power_mode=0
    # fi
else
    echo "v7"
    adb -s ${1} shell "cd $BIN_PATH && ./test_model_bin_v7 --model_dir=./shoubai_cpu/${2} --input_shape=${4} --out_txt=${3} --warmup=${5} --repeats=${6} --threads=${7} --use_optimize_nb=true --power_mode=3 && exit" #
    # if [ ${1} == '17c3cc34' ]; then
    #     adb -s ${1} shell "cd $BIN_PATH && ./test_model_bin_v7 --model_dir=./model/${2} --input_shape=${4} --out_txt=${3} --warmup=${5} --repeats=${6} --threads=${7} --use_optimize_nb=true --power_mode=3 && exit" #
    # else
    #     adb -s ${1} shell "cd $BIN_PATH && ./test_model_bin_v7 --model_dir=./model/${2} --input_shape=${4} --out_txt=${3} --warmup=${5} --repeats=${6} --threads=${7} --use_optimize_nb=true  && exit" #--power_mode=0
    # fi
fi 

adb -s ${1} pull $BIN_PATH/${3} $RESULT_ROOT

# ./test_model_bin_v7 --model_dir=./model/automl_mv3_5ms_64_s_ftdongxiao_shoubai/ --input_shape=1,3,64,64 --out_txt=out.txt --warmup=10 --repeats=50 --threads=1 --power_mode=0 --use_optimize_nb=1
# ./test_model_bin_v8 --model_dir=./model/automl_mv3_5ms_64_s_ftdongxiao_shoubai/ --input_shape=1,3,64,64 --out_txt=out.txt --warmup=10 --repeats=50 --threads=1 --power_mode=0 --use_optimize_nb=1

# ./test_model_bin_v7 --model_dir=./model/ar_cup_detection_int8/detection/ --input_shape=1,3,192,192 --out_txt=out.txt --warmup=10 --repeats=50 --threads=1 --power_mode=0 --use_optimize_nb=1
# ./test_model_bin_v8 --model_dir=./model/ar_cup_detection_int8/detection/ --input_shape=1,3,192,192 --out_txt=out.txt --warmup=10 --repeats=50 --threads=1 --power_mode=0 --use_optimize_nb=1

# ./test_model_bin_v7 --model_dir=./model/ar_cup_detection_int8/track/ --input_shape=1,3,128,128 --out_txt=out.txt --warmup=10 --repeats=50 --threads=1 --power_mode=0 --use_optimize_nb=1
# ./test_model_bin_v8 --model_dir=./model/ar_cup_detection_int8/track/ --input_shape=1,3,128,128 --out_txt=out.txt --warmup=10 --repeats=50 --threads=1 --power_mode=0 --use_optimize_nb=1

# ./test_model_bin_v7 --model_dir=./model/eye_mv1s_infer/ --input_shape=1,3,24,24 --out_txt=out.txt --warmup=10 --repeats=50 --threads=1 --power_mode=0 --use_optimize_nb=1
# ./test_model_bin_v8 --model_dir=./model/eye_mv1s_infer/ --input_shape=1,3,24,24 --out_txt=out.txt --warmup=10 --repeats=50 --threads=1 --power_mode=0 --use_optimize_nb=1

# ./test_model_bin_v7 --model_dir=./model/int8/ --input_shape=1,3,512,512 --out_txt=out.txt --warmup=10 --repeats=50 --threads=1 --power_mode=0 --use_optimize_nb=1
# ./test_model_bin_v8 --model_dir=./model/int8/ --input_shape=1,3,512,512 --out_txt=out.txt --warmup=10 --repeats=50 --threads=1 --power_mode=0 --use_optimize_nb=1

# ./test_model_bin_v7 --model_dir=./model/handkeypoints/kpt_model_detection/ --input_shape=1,3,224,224 --out_txt=out.txt --warmup=10 --repeats=50 --threads=1 --power_mode=0 --use_optimize_nb=1
# ./test_model_bin_v8 --model_dir=./model/handkeypoints/kpt_model_detection/ --input_shape=1,3,224,224 --out_txt=out.txt --warmup=10 --repeats=50 --threads=1 --power_mode=0 --use_optimize_nb=1

# ./test_model_bin_v7 --model_dir=./model/handkeypoints/kpt_model_keypoints/ --input_shape=1,3,144,256 --out_txt=out.txt --warmup=10 --repeats=50 --threads=1 --power_mode=0 --use_optimize_nb=1
# ./test_model_bin_v8 --model_dir=./model/handkeypoints/kpt_model_keypoints/ --input_shape=1,3,144,256 --out_txt=out.txt --warmup=10 --repeats=50 --threads=1 --power_mode=0 --use_optimize_nb=1

# ./test_model_bin_v7 --model_dir=./model/merge21_ssd_shufflenet_quant-fluild/ --input_shape=1,3,256,256 --out_txt=out.txt --warmup=10 --repeats=50 --threads=1 --power_mode=0 --use_optimize_nb=1
# ./test_model_bin_v8 --model_dir=./model/merge21_ssd_shufflenet_quant-fluild/ --input_shape=1,3,256,256 --out_txt=out.txt --warmup=10 --repeats=50 --threads=1 --power_mode=0 --use_optimize_nb=1

# ./test_model_bin_v7 --model_dir=./model/merge21-sbl-shufflenet-fluid/ --input_shape=1,3,192,192 --out_txt=out.txt --warmup=10 --repeats=50 --threads=1 --power_mode=0 --use_optimize_nb=1
# ./test_model_bin_v8 --model_dir=./model/merge21-sbl-shufflenet-fluid/ --input_shape=1,3,192,192 --out_txt=out.txt --warmup=10 --repeats=50 --threads=1 --power_mode=0 --use_optimize_nb=1

# ./test_model_bin_v7 --model_dir=./model/models_0158/ --input_shape=1,3,224,128 --out_txt=out.txt --warmup=10 --repeats=50 --threads=1 --power_mode=0 --use_optimize_nb=1
# ./test_model_bin_v8 --model_dir=./model/models_0158/ --input_shape=1,3,224,128 --out_txt=out.txt --warmup=10 --repeats=50 --threads=1 --power_mode=0 --use_optimize_nb=1

# ./test_model_bin_v7 --model_dir=./model/mouth_mv6_epoch320_shoubai/ --input_shape=1,3,48,48 --out_txt=out.txt --warmup=10 --repeats=50 --threads=1 --power_mode=0 --use_optimize_nb=1
# ./test_model_bin_v8 --model_dir=./model/mouth_mv6_epoch320_shoubai/ --input_shape=1,3,48,48 --out_txt=out.txt --warmup=10 --repeats=50 --threads=1 --power_mode=0 --use_optimize_nb=1

# ./test_model_bin_v7 --model_dir=./model/mv3_gp_shoubai/ --input_shape=1,3,128,128 --out_txt=out.txt --warmup=10 --repeats=50 --threads=1 --power_mode=0 --use_optimize_nb=1
# ./test_model_bin_v8 --model_dir=./model/mv3_gp_shoubai/ --input_shape=1,3,128,128 --out_txt=out.txt --warmup=10 --repeats=50 --threads=1 --power_mode=0 --use_optimize_nb=1

# ./test_model_bin_v7 --model_dir=./model/mv8_angle_shoubai/ --input_shape=1,3,64,64 --out_txt=out.txt --warmup=10 --repeats=50 --threads=1 --power_mode=0 --use_optimize_nb=1
# ./test_model_bin_v8 --model_dir=./model/mv8_angle_shoubai/ --input_shape=1,3,64,64 --out_txt=out.txt --warmup=10 --repeats=50 --threads=1 --power_mode=0 --use_optimize_nb=1

# ./test_model_bin_v7 --model_dir=./model/skyseg_shufflenet_0520_160/ --input_shape=1,3,160,160 --out_txt=out.txt --warmup=10 --repeats=50 --threads=1 --power_mode=0 --use_optimize_nb=1
# ./test_model_bin_v8 --model_dir=./model/skyseg_shufflenet_0520_160/ -input_shape=1,3,160,160 --out_txt=out.txt --warmup=10 --repeats=50 --threads=1 --power_mode=0 --use_optimize_nb=1

# ./test_model_bin_v7 --model_dir=./model/paddle_pose_merged_models/v1_7_person_det_merged_paddle/ --input_shape=1,3,128,128 --out_txt=out.txt --warmup=10 --repeats=50 --threads=1 --power_mode=0 --use_optimize_nb=1
# ./test_model_bin_v8 --model_dir=./model/paddle_pose_merged_models/v1_7_person_det_merged_paddle/ -input_shape=1,3,128,128 --out_txt=out.txt --warmup=10 --repeats=50 --threads=1 --power_mode=0 --use_optimize_nb=1

# ./test_model_bin_v7 --model_dir=./model/paddle_pose_merged_models/v2_4_2dpose_anakin_merged_paddle/ --input_shape=1,3,192,144 --out_txt=out.txt --warmup=10 --repeats=50 --threads=1 --power_mode=0 --use_optimize_nb=1
# ./test_model_bin_v8 --model_dir=./model/paddle_pose_merged_models/v2_4_2dpose_anakin_merged_paddle/ -input_shape=1,3,192,144 --out_txt=out.txt --warmup=10 --repeats=50 --threads=1 --power_mode=0 --use_optimize_nb=1

# ./test_model_bin_v7 --model_dir=./model/paddle_pose_merged_models/v2_6_2dpose_dlc_merged_paddle/ --input_shape=1,3,192,144 --out_txt=out.txt --warmup=10 --repeats=50 --threads=1 --power_mode=0 --use_optimize_nb=1
# ./test_model_bin_v8 --model_dir=./model/paddle_pose_merged_models/v2_6_2dpose_dlc_merged_paddle/ -input_shape=1,3,192,144 --out_txt=out.txt --warmup=10 --repeats=50 --threads=1 --power_mode=0 --use_optimize_nb=1

# ./test_model_bin_v7 --model_dir=./model/paddle_gesture_merged_models/v1_fingertip_pose_merge_paddle/ --input_shape=1,3,112,112 --out_txt=out.txt --warmup=10 --repeats=50 --threads=1 --power_mode=0 --use_optimize_nb=1
# ./test_model_bin_v8 --model_dir=./model/paddle_gesture_merged_models/v1_fingertip_pose_merge_paddle/ -input_shape=1,3,112,112 --out_txt=out.txt --warmup=10 --repeats=50 --threads=1 --power_mode=0 --use_optimize_nb=1

# ./test_model_bin_v7 --model_dir=./model/paddle_gesture_merged_models/v2_2_gesture_cls_merged_paddle/ --input_shape=1,3,96,96 --out_txt=out.txt --warmup=10 --repeats=50 --threads=1 --power_mode=0 --use_optimize_nb=1
# ./test_model_bin_v8 --model_dir=./model/paddle_gesture_merged_models/v2_2_gesture_cls_merged_paddle/ -input_shape=1,3,96,96 --out_txt=out.txt --warmup=10 --repeats=50 --threads=1 --power_mode=0 --use_optimize_nb=1

# ./test_model_bin_v7 --model_dir=./model/paddle_gesture_merged_models/v5_3_5_gesture_det_merged_paddle/ --input_shape=1,3,128,128 --out_txt=out.txt --warmup=10 --repeats=50 --threads=1 --power_mode=0 --use_optimize_nb=1
# ./test_model_bin_v8 --model_dir=./model/paddle_gesture_merged_models/v5_3_5_gesture_det_merged_paddle/ -input_shape=1,3,128,128 --out_txt=out.txt --warmup=10 --repeats=50 --threads=1 --power_mode=0 --use_optimize_nb=1
