
rm -rf build
mkdir build
cd build

LITE_ROOT=$1
NDK_ROOT=$2


cmake .. \
         -DLITE_ROOT=${LITE_ROOT} \
         -DNDK_ROOT=${NDK_ROOT} \
         -DCMAKE_TOOLCHAIN_FILE=${NDK_ROOT}/build/cmake/android.toolchain.cmake \
         -DANDROID_TOOLCHAIN=gcc \
         -DANDROID_ABI="armeabi-v7a" \
         -DANDROID_PLATFORM=android-23 \
         -DANDROID=true \
         -DANDROID_STL=c++_static
make
cd ..
# ./bin/demo_trainer
