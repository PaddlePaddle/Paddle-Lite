export LD_LIBRARY_PATH=$PWD/lite/lib/:${LD_LIBRARY_PATH}

# mobilenetv1
model_name="mobilenetv1"
input_params="--img_txt_path=models_imgs/images/classification.jpg.txt \
              --out_max_value=0.936887 \
              --out_max_value_index=65"
echo "Test ${model_name}: light_shared, light_static, full_shared, full_static."

./classification_light_shared ${input_params} \
    --optimized_model_path=models_imgs/models/mobilenetv1.nb

./classification_light_static ${input_params} \
    --optimized_model_path=models_imgs/models/mobilenetv1.nb

./classification_full_shared ${input_params} \
    --model_dir=models_imgs/models/mobilenetv1

./classification_full_static ${input_params} \
    --model_dir=models_imgs/models/mobilenetv1

# mobilenetv2
model_name="mobilenetv2"
input_params="--img_txt_path=models_imgs/images/classification.jpg.txt \
              --out_max_value=0.868888 \
              --out_max_value_index=65"
echo "Test ${model_name}: light_shared, light_static, full_shared, full_static."

./classification_light_shared ${input_params} \
    --optimized_model_path=models_imgs/models/mobilenetv2.nb

./classification_light_static ${input_params} \
    --optimized_model_path=models_imgs/models/mobilenetv2.nb

./classification_full_shared ${input_params} \
    --model_dir=models_imgs/models/mobilenetv2

./classification_full_static ${input_params} \
    --model_dir=models_imgs/models/mobilenetv2

# shufflenetv2
model_name="shufflenetv2"
input_params="--img_txt_path=models_imgs/images/classification.jpg.txt \
              --out_max_value=0.776729 \
              --out_max_value_index=65"
echo "Test ${model_name}: light_shared, light_static, full_shared, full_static."

./classification_light_shared ${input_params} \
    --optimized_model_path=models_imgs/models/shufflenetv2.nb

./classification_light_static ${input_params} \
    --optimized_model_path=models_imgs/models/shufflenetv2.nb

./classification_full_shared ${input_params} \
    --model_dir=models_imgs/models/shufflenetv2

./classification_full_static ${input_params} \
    --model_dir=models_imgs/models/shufflenetv2

# yolov3
model_name="yolov3"
input_params="--img_txt_path=models_imgs/images/yolov3.jpg.txt \
              --out_values=0,0.153605,174.494,199.729,562.075,604.014"
echo "Test ${model_name}: light_shared, light_static, full_shared, full_static."

./yolov3_light_shared ${input_params} \
    --optimized_model_path=models_imgs/models/yolov3_mobilenetv1.nb

./yolov3_light_static ${input_params} \
    --optimized_model_path=models_imgs/models/yolov3_mobilenetv1.nb

./yolov3_full_shared ${input_params} \
    --model_dir=models_imgs/models/yolov3_mobilenetv1

./yolov3_full_static ${input_params} \
    --model_dir=models_imgs/models/yolov3_mobilenetv1
