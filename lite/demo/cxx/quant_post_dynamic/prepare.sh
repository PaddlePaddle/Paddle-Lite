make all -j

gf=quant_post_dynamic_demo
if [ -d ${gf} ];then
    rm -rf ${gf}
fi
mkdir ${gf}

wget https://paddle-inference-dist.cdn.bcebos.com/PaddleLite/models_and_data_for_unittests/imgnet_val_1_jpg_txt.tar.gz
tar zxf imgnet_val_1_jpg_txt.tar.gz

cp classification_light_shared ${gf}
cp run.sh ${gf}
cp ../../../cxx/lib/libpaddle_light_api_shared.so ${gf}
cp imgnet_val_1_jpg_txt ${gf}

make clean
