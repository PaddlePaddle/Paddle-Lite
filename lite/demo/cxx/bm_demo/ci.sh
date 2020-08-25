wget http://paddle-inference-dist.bj.bcebos.com/mobilenet_v1.tar.gz
tar -xvf mobilenet_v1.tar.gz

mkdir ./lib
cp -r /paddlelite/build.lite.bm/lite/api/libpaddle_full_api_shared.so \
/paddlelite/third-party/bmlibs/bm_sc3_libs/lib/bmcompiler \
/paddlelite/third-party/bmlibs/bm_sc3_libs/lib/bmnn/pcie ./lib

mkdir ./include
cp -r /paddlelite/lite/api/paddle_place.h /paddlelite/lite/api/paddle_api.h ./include

mkdir ./build
cd ./build
cmake ..
make
cd ..
rm -rf ./build
rm -rf ./mobilenet_v1.tar.gz

./mobilenet_full_api ./mobilenet_v1 224 224
