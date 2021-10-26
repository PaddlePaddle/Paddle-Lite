# setting NDK_ROOT root
export NDK_ROOT=/opt/android-ndk-r20b

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

# v2.10_rc model, if need other version model, such as v2.9.1, you can use the following command:
# wget -c https://paddlelite-demo.bj.bcebos.com/models/pyramidbox_lite_fp32_for_cpu_v2_9_1.tar.gz
# other version model just change string of "v2_9_1" to responding string
if [ ! -f "pyramidbox_lite_fp32_for_cpu_v2_10_rc.tar.gz" ];
then
wget -c https://paddlelite-demo.bj.bcebos.com/models/pyramidbox_lite_fp32_for_cpu_v2_10_rc.tar.gz
fi

# v2.10_rc model
if [ ! -f "mask_detector_fp32_128_128_for_cpu_v2_10_rc.tar.gz" ];
then
wget -c https://paddlelite-demo.bj.bcebos.com/models/mask_detector_fp32_128_128_for_cpu_v2_10_rc.tar.gz
fi

tar zxf mask_models_img.tar.gz
tar zxf pyramidbox_lite_fp32_for_cpu_v2_10_rc.tar.gz
mv model.nb pyramidbox_lite_v2_10_rc_opt2.nb
tar zxf mask_detector_fp32_128_128_for_cpu_v2_10_rc.tar.gz
mv model.nb mask_detector_v2_10_rc_opt2.nb
mv mask_models_img ${gf}
mv pyramidbox_lite_v2_10_rc_opt2.nb ${gf}
mv mask_detector_v2_10_rc_opt2.nb ${gf}

# clean
make clean
