export LD_LIBRARY_PATH=$PWD:${LD_LIBRARY_PATH}

# test mobilenetv1

input_params="--img_txt_path=./imgnet_val_1_jpg_txt \
              --out_max_value=0.936887 \
              --out_max_value_index=65 \
              --threshold=0.01"

./classification_light_shared ${input_params} \
    --optimized_model_path=./mobilenet_v1_opt.nb
echo "Finish test mobilenet_v1_opt\n\n"

./classification_light_shared ${input_params} \
    --optimized_model_path=./mobilenet_v1_int16_opt.nb
echo "Finish test mobilenet_v1_int16_opt\n\n"

./classification_light_shared ${input_params} \
    --optimized_model_path=./mobilenet_v1_int8_opt.nb
echo "Finish test mobilenet_v1_int8_opt\n\n"
