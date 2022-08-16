
export NDK_ROOT=/opt/android-ndk-r20b
export PATH=$NDK_ROOT:$PATH

wget https://paddlelite-demo.bj.bcebos.com/demo/ernie_faster_tokenizer/faster_tokenizer.tar.gz
tar -zxvf faster_tokenizer.tar.gz
wget https://paddlelite-demo.bj.bcebos.com/demo/ernie_faster_tokenizer/assets.tar.gz
tar -zxvf assets.tar.gz

rm -rf *.tar.gz*

mkdir lite
cp -r ../../../cxx ./lite

rm -rf build && mkdir build && cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=$NDK_ROOT/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI="arm64-v8a" \
  -DANDROID_NATIVE_API_LEVEL=android-21 \
  -DANDROID_STL=c++_static

make -j8
