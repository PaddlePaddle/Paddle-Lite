make clean
make all -j

gf=test_lite_lib_files
if [ -d ${gf} ];then
    rm -rf ${gf}
fi
mkdir ${gf}

mv classification_full_shared ${gf}
mv classification_full_static ${gf}
mv classification_light_shared ${gf}
mv classification_light_static ${gf}
mv yolov3_full_shared ${gf}
mv yolov3_full_static ${gf}
mv yolov3_light_shared ${gf}
mv yolov3_light_static ${gf}
cp run.sh ${gf}

make clean

cp -r ../../../cxx/ ${gf}
mv ${gf}/cxx ${gf}/lite

if [ ! -f "test_libs_models_imgs.tgz" ];then
    wget https://paddle-inference-dist.cdn.bcebos.com/PaddleLite/test_libs_models_imgs.tgz
fi
tar zxf test_libs_models_imgs.tgz
mv test_libs_models_imgs ${gf}
mv ${gf}/test_libs_models_imgs ${gf}/models_imgs
