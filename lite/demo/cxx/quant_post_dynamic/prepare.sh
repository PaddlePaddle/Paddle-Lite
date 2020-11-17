make all -j

gf=quant_post_dynamic_demo
if [ -d ${gf} ];then
    rm -rf ${gf}
fi
mkdir ${gf}

cp classification_light_shared ${gf}
cp run.sh ${gf}
cp ../../../cxx/lib/libpaddle_light_api_shared.so ${gf}

make clean
