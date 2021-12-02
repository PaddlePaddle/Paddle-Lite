#!/bin/bash
set -e
set -x

## Global variables
workspace=$PWD/$(dirname $0)
readonly workspace=${workspace%%lite/tools*}
WITH_LOG=OFF
WITH_CV=ON
WITH_EXCEPTION=ON
# ToolChain options: clang gcc
TOOL_CHAIN=clang
# AndroidStl options: c++_static c++_shared
ANDROID_STL=c++_static

## step 1: compile opt tool
cd $workspace
if [ ! -f build.opt/lite/api/opt ]; then
./lite/tools/build.sh build_optimize_tool
fi
cd build.opt/lite/api
rm -rf models &&  cp -rf $1 ./models

###  models names
models_names=$(ls models)
## step 2. convert models
rm -rf models_opt && mkdir models_opt
for name in $models_names
do
  ./opt --model_dir=./models/$name --valid_targets=arm --optimize_out=./models_opt/$name --record_tailoring_info=true
done


# step 3. record model infos
rm -rf model_info && mkdir model_info
rm -rf optimized_model && mkdir optimized_model
content=$(ls ./models_opt | grep -v .nb)

for dir_name in $content
do
cat ./models_opt/$dir_name/.tailored_kernels_list >> ./model_info/tailored_kernels_list 
cat ./models_opt/$dir_name/.tailored_kernels_source_list >> ./model_info/tailored_kernels_source_list
cat ./models_opt/$dir_name/.tailored_ops_list >> ./model_info/tailored_ops_list
cat ./models_opt/$dir_name/.tailored_ops_source_list >> ./model_info/tailored_ops_source_list
cp -f ./models_opt/$dir_name.nb optimized_model
done

cp ./model_info/tailored_kernels_list ./model_info/.tailored_kernels_list
cp ./model_info/tailored_kernels_source_list ./model_info/.tailored_kernels_source_list
cp ./model_info/tailored_ops_list ./model_info/.tailored_ops_list
cp ./model_info/tailored_ops_source_list ./model_info/.tailored_ops_source_list
rm -rf $(ls ./models_opt | grep -v .nb)

# step 4. compiling Android ARM lib
cd $workspace
./lite/tools/build_android.sh --with_strip=ON --opt_model_dir=$workspace/build.opt/lite/api/model_info --with_log=$WITH_LOG --with_cv=$WITH_CV --toolchain=$TOOL_CHAIN --with_exception=$WITH_EXCEPTION --android_stl=$ANDROID_STL

./lite/tools/build_android.sh --with_strip=ON --opt_model_dir=$workspace/build.opt/lite/api/model_info --with_log=$WITH_LOG --arch=armv7 --with_cv=$WITH_CV --toolchain=$TOOL_CHAIN --with_exception=$WITH_EXCEPTION --android_stl=$ANDROID_STL

# step 5. pack compiling results and optimized models
result_name=android_lib
rm -rf $result_name && mkdir $result_name
cp -rf build.lite.android.armv7.$TOOL_CHAIN/inference_lite_lib.android.armv7 $result_name/armv7.$TOOL_CHAIN
cp -rf build.lite.android.armv8.$TOOL_CHAIN/inference_lite_lib.android.armv8 $result_name/armv8.$TOOL_CHAIN
cp build.opt/lite/api/opt $result_name/
mv build.opt/lite/api/optimized_model $result_name

# step6. compress the result into tar file
tar zcf $result_name.tar.gz $result_name
