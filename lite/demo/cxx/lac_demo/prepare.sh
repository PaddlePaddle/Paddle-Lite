# make
make -j

# mkdir
gf=lac_demo_file
if [ -d ${gf} ];then
    rm -rf ${gf}
fi
mkdir ${gf}

# collect files
cp run.sh ${gf}
cp lac_demo ${gf}
cp ../../../cxx/lib/libpaddle_light_api_shared.so ${gf}

if [ ! -f "lac_model_conf_data.tgz" ];
then
   wget -c https://paddle-inference-dist.cdn.bcebos.com/PaddleLite/models_and_data_for_unittests/lac_model_conf_data.tgz
fi
tar zxf lac_model_conf_data.tgz
mv lac_model_conf_data ${gf}

# clean
make clean
