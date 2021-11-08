#!/bin/bash
set +x
set -e

# Absolute path of Paddle-Lite source code.
SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
WORKSPACE=${SHELL_FOLDER%tools/ci_tools*}
NUM_PROC=4

# model name and downloading url
MODELS_URL=( "http://paddle-inference-dist.bj.bcebos.com/mobilenet_v1.tar.gz" \  # mobilenet_v1 
           )


# if operating in mac env, we should expand the maximum file num
os_name=`uname -s`
if [ ${os_name} == "Darwin" ]; then
   ulimit -n 1024
fi

####################################################################################################
# 1. functions of prepare workspace before compiling
####################################################################################################

# 1.1 download models into `inference_model` directory 
function prepare_models {
  cd ${WORKSPACE}
  rm -rf inference_model && mkdir inference_model && cd inference_model
  # download compressed model recorded in $MODELS_URL
  for url in ${MODELS_URL[@]}; do
    wget $url
  done

  compressed_models=$(ls)
  # decompress models
  for name in ${compressed_models[@]}; do
    if echo "$name" | grep -q -E '.tar.gz$'; then
      tar xf $name && rm -f $name
    elif echo "$name" | grep -q -E '.zip$'; then
      unzip $name && rm -f $name
    else
      echo "Error, only .zip or .tar.gz format files are supported!"
    fi
  done
  cd ${WORKSPACE}
}

####################################################################################################
# 2. functions of compiling stripped libs according to models in `inference_model`
####################################################################################################

# 2.1 Compile the stripped lib and transform the inputed model, results will be stored into `android_lib`
#    `android_lib`
#         |---------armv7.clang
#         |---------armv8.clang
#         |---------optimized_model
#         |---------opt
function compile_according_to_models {
  cd ${WORKSPACE}
  ./lite/tools/build_android_by_models.sh ${WORKSPACE}/inference_model
}

####################################################################################################
# 3. functions of testing model on android platform.
####################################################################################################

# 3.1 Compile Android demo and upload them onto adb device to perform unit test
function test_model {
  adb_index=$1
  adb_dir=$2

  adb_devices=($(adb devices |grep -v devices |grep device | awk -F " " '{print $1}'))

  # prepare workspace directory
  adb -s ${adb_devices[$adb_index]} shell "cd /data/local/tmp && rm -rf $adb_dir && mkdir $adb_dir"
  # 1. upload optimized model
  adb -s ${adb_devices[$adb_index]} push android_lib/optimized_model/mobilenet_v1.nb /data/local/tmp/$adb_dir

  # 2. perform armv7 unit_test
  #    2.1 upload armv7 lib
  adb -s ${adb_devices[$adb_index]} push android_lib/armv7.clang/cxx/lib/libpaddle_light_api_shared.so /data/local/tmp/$adb_dir
  #    2.2 compile and upload armv7 demo
  cd android_lib/armv7.clang/demo/cxx/mobile_light && make && chmod +x mobilenetv1_light_api
  adb -s ${adb_devices[$adb_index]} push mobilenetv1_light_api /data/local/tmp/$adb_dir  && cd -
  #    2.3 perform unit test
  adb -s ${adb_devices[$adb_index]} shell "cd /data/local/tmp/$adb_dir && export LD_LIBRARY_PATH=./ &&  ./mobilenetv1_light_api ./mobilenet_v1.nb"

  # 3. perform armv8 unit_test
  #    3.1 upload armv8 lib
  adb -s ${adb_devices[$adb_index]} push android_lib/armv8.clang/cxx/lib/libpaddle_light_api_shared.so /data/local/tmp/$adb_dir
  #    3.2 compile and upload armv8 demo
  cd android_lib/armv8.clang/demo/cxx/mobile_light && make && chmod +x mobilenetv1_light_api
  adb -s ${adb_devices[$adb_index]} push mobilenetv1_light_api /data/local/tmp/$adb_dir  && cd -
  #    3.3 perform unit test
  adb -s ${adb_devices[$adb_index]} shell "cd /data/local/tmp/$adb_dir && export LD_LIBRARY_PATH=./ &&  ./mobilenetv1_light_api ./mobilenet_v1.nb"
}

# 3.2 Compile Android FP16 demo and upload them onto adb device to perform unit test
function test_fp16_model {
  cd ${WORKSPACE}
  workspace=$PWD/$(dirname $0)
  readonly workspace=${workspace%%lite/tools*}
  WITH_LOG=OFF
  WITH_CV=ON
  WITH_EXCEPTION=ON
  # ToolChain options: clang gcc
  TOOL_CHAIN=clang
  # AndroidStl options: c++_static c++_shared
  ANDROID_STL=c++_static
  cd $workspace
  # step 1 build opt tool
  if [ ! -f build.opt/lite/api/opt ]; then
  ./lite/tools/build.sh build_optimize_tool
  fi
  ###  models names
  rm -rf fp16_models &&  cp -rf ./inference_model ./fp16_models
  models_names=$(ls fp16_models)
  ## step 2. convert models
  rm -rf optimized_model_fp16 && mkdir optimized_model_fp16
  for name in $models_names
  do
    ./build.opt/lite/api/opt --model_dir=./models/$name --valid_targets=arm --optimize_out=./optimized_model_fp16/$name --enable_fp16=1
  done

  # step 3. compiling Android ARM lib
  export NDK_ROOT=/opt/android-ndk-r23b/
  if [ -f build.lite.android.armv7.clang ]; then
    rm -rf build.lite.android.armv7.clang
  fi
  if [ -f build.lite.android.armv8.clang ]; then
    rm -rf build.lite.android.armv8.clang
  fi
  ./lite/tools/build_android.sh --with_strip=ON --with_log=$WITH_LOG --with_cv=$WITH_CV --toolchain=$TOOL_CHAIN --with_exception=$WITH_EXCEPTION --android_stl=$ANDROID_STL --build_arm82_fp16=ON

  ./lite/tools/build_android.sh --with_strip=ON --with_log=$WITH_LOG --arch=armv7 --with_cv=$WITH_CV --toolchain=$TOOL_CHAIN --with_exception=$WITH_EXCEPTION --android_stl=$ANDROID_STL --build_arm82_fp16=ON

  # step 4. pack compiling results and optimized models
  result_name=android_lib_fp16
  rm -rf $result_name && mkdir $result_name
  cp -rf build.lite.android.armv7.$TOOL_CHAIN/inference_lite_lib.android.armv7 $result_name/armv7_fp16.$TOOL_CHAIN
  cp -rf build.lite.android.armv8.$TOOL_CHAIN/inference_lite_lib.android.armv8 $result_name/armv8_fp16.$TOOL_CHAIN
  cp build.opt/lite/api/opt $result_name/
  mv optimized_model_fp16 $result_name

  # step 5. run ut
  adb_index=$1
  adb_dir=$2

  adb_devices=($(adb devices |grep -v devices |grep device | awk -F " " '{print $1}'))

  # prepare workspace directory
  adb -s ${adb_devices[$adb_index]} shell "cd /data/local/tmp && rm -rf $adb_dir && mkdir $adb_dir"
  # 5.1. upload optimized model
  adb -s ${adb_devices[$adb_index]} push android_lib_fp16/optimized_model_fp16/mobilenet_v1.nb /data/local/tmp/$adb_dir

  # 5.2. perform armv7 unit_test v7-ut doesn't support
  #    5.2.1 upload armv7 lib
  # adb -s ${adb_devices[$adb_index]} push android_lib_fp16/armv7_fp16.clang/cxx/lib/libpaddle_light_api_shared.so /data/local/tmp/$adb_dir
  #    5.2.2 compile and upload armv7 demo
  # cd android_lib_fp16/armv7_fp16.clang/demo/cxx/mobile_light && make && chmod +x mobilenetv1_light_api
  # adb -s ${adb_devices[$adb_index]} push mobilenetv1_light_api /data/local/tmp/$adb_dir  && cd -
  #    5.2.3 perform unit test
  # adb -s ${adb_devices[$adb_index]} shell "cd /data/local/tmp/$adb_dir && export LD_LIBRARY_PATH=./ &&  ./mobilenetv1_light_api ./mobilenet_v1.nb"

  # 5.3. perform armv8 unit_test
  #    5.3.1 upload armv8 lib
  adb -s ${adb_devices[$adb_index]} push android_lib_fp16/armv8_fp16.clang/cxx/lib/libpaddle_light_api_shared.so /data/local/tmp/$adb_dir
  #    5.3.2 compile and upload armv8 demo
  cd android_lib_fp16/armv8_fp16.clang/demo/cxx/mobile_light && make && chmod +x mobilenetv1_light_api
  adb -s ${adb_devices[$adb_index]} push mobilenetv1_light_api /data/local/tmp/$adb_dir  && cd -
  #    5.3.3 perform unit test
  adb -s ${adb_devices[$adb_index]} shell "cd /data/local/tmp/$adb_dir && export LD_LIBRARY_PATH=./ &&  ./mobilenetv1_light_api ./mobilenet_v1.nb"
}

####################################################################################################
# 4. main function
#    $1: adb device index, eg. 0 1
#    $2: adb workspace directory name
####################################################################################################

function main {
  prepare_models
  compile_according_to_models
  test_model $1 $2
  test_fp16_model $1 $2
}

main $1 $2
