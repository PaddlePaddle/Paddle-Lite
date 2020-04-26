export LD_LIBRARY_PATH=$PWD/lite/lib/:${LD_LIBRARY_PATH}

# mobilenetv1

./classification_light_shared \
    --optimized_model_path=models_imgs/models/mobilenetv1.nb \
    --img_txt_path=models_imgs/images/classification.jpg.txt \
    --out_max_value=0.936887 \
    --out_max_value_index=65

./classification_light_static \
    --optimized_model_path=models_imgs/models/mobilenetv1.nb \
    --img_txt_path=models_imgs/images/classification.jpg.txt \
    --out_max_value=0.936887 \
    --out_max_value_index=65

./classification_full_static \
    --model_dir=models_imgs/models/mobilenetv1 \
    --img_txt_path=models_imgs/images/classification.jpg.txt \
    --out_max_value=0.936887 \
    --out_max_value_index=65

./classification_full_shared \
    --model_dir=models_imgs/models/mobilenetv1 \
    --img_txt_path=models_imgs/images/classification.jpg.txt \
    --out_max_value=0.936887 \
    --out_max_value_index=65

# mobilenetv2

./classification_light_shared \
    --optimized_model_path=models_imgs/models/mobilenetv2.nb \
    --img_txt_path=models_imgs/images/classification.jpg.txt \
    --out_max_value=0.868888 \
    --out_max_value_index=65

./classification_light_static \
    --optimized_model_path=models_imgs/models/mobilenetv2.nb \
    --img_txt_path=models_imgs/images/classification.jpg.txt \
    --out_max_value=0.868888 \
    --out_max_value_index=65

./classification_full_static \
    --model_dir=models_imgs/models/mobilenetv2 \
    --img_txt_path=models_imgs/images/classification.jpg.txt \
    --out_max_value=0.868888 \
    --out_max_value_index=65

./classification_full_shared \
    --model_dir=models_imgs/models/mobilenetv2 \
    --img_txt_path=models_imgs/images/classification.jpg.txt \
    --out_max_value=0.868888 \
    --out_max_value_index=65

# yolov3

./yolov3_light_shared \
    --optimized_model_path=models_imgs/models/yolov3_mobilenetv1.nb  \
    --img_txt_path=models_imgs/images/yolov3.jpg.txt \
    --out_values=0,0.153605,174.494,199.729,562.075,604.014

./yolov3_light_static \
    --optimized_model_path=models_imgs/models/yolov3_mobilenetv1.nb \
    --img_txt_path=models_imgs/images/yolov3.jpg.txt \
    --out_values=0,0.153605,174.494,199.729,562.075,604.014

./yolov3_full_static \
    --model_dir=models_imgs/models/yolov3_mobilenetv1 \
    --img_txt_path=models_imgs/images/yolov3.jpg.txt \
    --out_values=0,0.153605,174.494,199.729,562.075,604.014

./yolov3_full_shared \
    --model_dir=models_imgs/models/yolov3_mobilenetv1 \
    --img_txt_path=models_imgs/images/yolov3.jpg.txt \
    --out_values=0,0.153605,174.494,199.729,562.075,604.014
