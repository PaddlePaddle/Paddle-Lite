# make
make -j

# mkdir
gf=mask_demo
if [ -d ${gf} ];then
    rm -rf ${gf}
fi
mkdir ${gf}

# collect files
cp run.sh ${gf}
cp mask_detection ${gf}
cp ../../../cxx/lib/libpaddle_light_api_shared.so ${gf}

if [ ! -f "mask_models_img.tar.gz" ];
then
   wget -c https://paddle-inference-dist.cdn.bcebos.com/PaddleLiteDemo/mask_models_img.tar.gz 
fi
tar zxf mask_models_img.tar.gz
mv mask_models_img ${gf}

# clean
make clean
