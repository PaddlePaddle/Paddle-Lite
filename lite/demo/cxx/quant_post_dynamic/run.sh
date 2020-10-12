export LD_LIBRARY_PATH=$PWD:${LD_LIBRARY_PATH}

# test mobilenetv1

input_params="--img_txt_path=./quant_post_dynamic/imgnet_val_1.jpg.txt \
              --out_max_value=0.936887 \
              --out_max_value_index=65"

./classification_light_shared ${input_params} \
    --optimized_model_path=./quant_post_dynamic/mobilenet_v1_opt.nb

./classification_light_shared ${input_params} \
    --optimized_model_path=./quant_post_dynamic/mobilenet_v1_int16_opt.nb
