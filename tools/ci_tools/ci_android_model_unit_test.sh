#!/bin/bash
set -x
set -e

# Absolute path of Paddle-Lite source code.
SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
WORKSPACE=${SHELL_FOLDER%tools/ci_tools*}
NUM_PROC=4

# model name and downloading url
MODELS_URL=( "https://paddlelite-data.bj.bcebos.com/arm/airank_paddlelite_source_model.tar.gz" \ )

# fluid output and downloading url
FLUID_OUTPUT_URL="https://paddlelite-data.bj.bcebos.com/arm/airank_fluid_output.tar.gz"

# if operating in mac env, we should expand the maximum file num
os_name=`uname -s`
if [ ${os_name} == "Darwin" ]; then
   ulimit -n 1024
fi

model_str=`python3 ci_model_unit_test.py modellist`
model_arr=(${model_str//./ })

shape_str=`python3 ci_model_unit_test.py shapelist`
shape_arr=(${shape_str//./ })

exclude_str=`python3 ci_model_unit_test.py excludelist`
exclude_arr=(${exclude_str//./ })

####################################################################################################
# 1. functions of prepare workspace before compiling
####################################################################################################

# 1.1 download models into `inference_model` directory 
function prepare_models {
  cd ${WORKSPACE}

  # download fluid output and decompress fluid output
  wget $FLUID_OUTPUT_URL
  fluid_output_tar_gz_tmp=(${FLUID_OUTPUT_URL//\// })
  fluid_output_tar_gz=${fluid_output_tar_gz_tmp[${#fluid_output_tar_gz_tmp[@]}-1]}
  fluid_output_tar_gz_out=(${fluid_output_tar_gz//./ })
  rm -rf $fluid_output_tar_gz_out && mkdir $fluid_output_tar_gz_out
  tar xf $fluid_output_tar_gz -C $fluid_output_tar_gz_out

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
function compile_according_lib_and_opt {
  cd ${WORKSPACE}  
  WITH_LOG=OFF
  WITH_CV=ON
  WITH_EXCEPTION=ON
  WITH_EXTRA=ON
  # ToolChain options: clang gcc
  TOOL_CHAIN=clang
  # AndroidStl options: c++_static c++_shared
  ANDROID_STL=c++_static

  # step 1: compile opt tool
  if [ ! -f build.opt/lite/api/opt ]; then
  ./lite/tools/build.sh build_optimize_tool
  fi

  cd build.opt/lite/api
  rm -rf models &&  cp -rf ${WORKSPACE}/inference_model ./models    
  ###  models names
  int8_models_names=$(ls models/paddlelite/int8)
  fp_models_names=$(ls models/paddlelite/fp32)
  ## step 2. convert models
  rm -rf optimized_model && mkdir optimized_model

  ## 2.1 convert int8 opt models  
  mkdir optimized_model/int8
  for name in $int8_models_names
  do
    param_name=""
    model_name="" 
    if [ -f ./models/paddlelite/int8/$name/inference.pdiparams ];then
      param_name=inference.pdiparams
      model_name=inference.pdmodel
    elif [ -f ./models/paddlelite/int8/$name/model.pdiparams ];then
      param_name=model.pdiparams
      model_name=model.pdmodel    
    elif [ -f ./models/paddlelite/int8/$name/__params__ ];then
      param_name=__params__
      model_name=__model__       
    fi  
    ./opt --model_file=./models/paddlelite/int8/$name/$model_name --param_file=./models/paddlelite/int8/$name/$param_name --valid_targets=arm --optimize_out=./optimized_model/int8/$name
  done

  ## 2.2 convert fp16/fp32 opt models  
  mkdir optimized_model/fp16 && mkdir optimized_model/fp32
  for name in $fp_models_names
  do
    param_name=""
    model_name="" 
    if [ -f ./models/paddlelite/fp32/$name/inference.pdiparams ];then
      param_name=inference.pdiparams
      model_name=inference.pdmodel
    elif [ -f ./models/paddlelite/fp32/$name/model.pdiparams ];then
      param_name=model.pdiparams
      model_name=model.pdmodel    
    elif [ -f ./models/paddlelite/fp32/$name/__params__ ];then
      param_name=__params__
      model_name=__model__       
    fi
      ./opt --model_file=./models/paddlelite/fp32/$name/$model_name --param_file=./models/paddlelite/fp32/$name/$param_name --valid_targets=arm --enable_fp16=true --optimize_out=./optimized_model/fp16/$name
      ./opt --model_file=./models/paddlelite/fp32/$name/$model_name --param_file=./models/paddlelite/fp32/$name/$param_name --valid_targets=arm --optimize_out=./optimized_model/fp32/$name
  done

  ## 2.3 calculate opt models name

  ## step 3. compiling Android ARM lib
  cd ${WORKSPACE} && rm -rf third_party
  ./lite/tools/build_android.sh --arch=armv8 --toolchain=$TOOL_CHAIN --android_stl=$ANDROID_STL --with_arm82_fp16=ON --with_benchmark=ON full_publish


  ./lite/tools/build_android.sh --arch=armv7 --toolchain=$TOOL_CHAIN --android_stl=$ANDROID_STL --with_arm82_fp16=ON --with_benchmark=ON full_publish


  cd build.lite.android.armv7.$TOOL_CHAIN/lite/api/tools/benchmark && mv benchmark_bin benchmark_bin_armv7 

  cd ${WORKSPACE}

  cd build.lite.android.armv8.$TOOL_CHAIN/lite/api/tools/benchmark && mv benchmark_bin benchmark_bin_armv8   

  # step 4. pack compiling results and optimized models
  cd ${WORKSPACE}  
  result_name=android_lib
  rm -rf $result_name && mkdir $result_name
  cp build.lite.android.armv7.$TOOL_CHAIN/lite/api/tools/benchmark/benchmark_bin_armv7 $result_name
  cp build.lite.android.armv8.$TOOL_CHAIN/lite/api/tools/benchmark/benchmark_bin_armv8 $result_name
  mv build.opt/lite/api/optimized_model $result_name

  # step5. compress the result into tar file
  tar zcf $result_name.tar.gz $result_name
}

####################################################################################################
# 3. functions of testing model on android platform.
####################################################################################################
function test_model {
  cd ${WORKSPACE}  
  adb_index=$1
  adb_dir=$2

  adb_devices=($(adb devices |grep -v devices |grep device | awk -F " " '{print $1}'))

  # 1. prepare

  # 1.1 prepare workspace directory
  adb -s ${adb_devices[$adb_index]} shell "cd /data/local/tmp && rm -rf $adb_dir && mkdir $adb_dir"

  # 1.2 upload optimized model and bench_mark_bin
  adb -s ${adb_devices[$adb_index]} push android_lib/optimized_model android_lib/benchmark_bin_armv8 android_lib/benchmark_bin_armv7 /data/local/tmp/$adb_dir

  # 2. model unit test

  #make output dir
  adb -s ${adb_devices[$adb_index]} shell " cd /data/local/tmp/$adb_dir && mkdir output && mkdir output/armv7 output/armv8 && mkdir output/armv7/int8 output/armv7/fp16 output/armv7/fp32 output/armv8/int8 output/armv8/fp16 output/armv8/fp32"

  accuracy_type=(fp16 fp32 int8)
  arm_abi=(armv8 armv7)
  for accuracy in ${accuracy_type[@]}; do
    #choose cpu precision   
    cpu_precision="fp32"
    if [ ${accuracy} == "fp16" ]; then
      cpu_precision="fp16"
    fi
    #update fp16/fp32 model name    
    for(( i=0;i<${#model_arr[@]};i++)); do
      if [ ${exclude_arr[$i]} == "True" ]; then
        continue
      fi
      model_accuracy[$i]=${model_arr[$i]}
    done
    #update int8 model name    
    if [ "$accuracy" = "int8" ]; then
      for(( i=0;i<${#model_accuracy[@]};i++)); do
        tmp=(${model_accuracy[$i]//_/ })
        model_accuracy[$i]=${model_accuracy[$i]/${tmp[${#tmp[@]}-1]}/"quant_"${tmp[${#tmp[@]}-1]}}
      done
    fi
    #run    
    for arm in ${arm_abi[@]}; do
      for(( i=0;i<${#model_accuracy[@]};i++ )); do
        opt_model=${model_accuracy[$i]}
        shape=${shape_arr[$i]}
        adb -s ${adb_devices[$adb_index]} shell "cd /data/local/tmp/$adb_dir &&  ./benchmark_bin_$arm --optimized_model_file=./optimized_model/$accuracy/$opt_model".nb" --backend=arm --input_shape=$shape --warmup=10 --repeats=10 --power_mode=0 --threads=1 --show_output_elem=true --cpu_precision=$cpu_precision > output/$arm/$accuracy/$opt_model".txt" 2>&1 "
      done
    done
  done
  
  adb -s ${adb_devices[$adb_index]} pull  /data/local/tmp/$adb_dir/output .
  cp -rf output output_bak
  adb -s ${adb_devices[$adb_index]} shell "rm -rf /data/local/tmp/$adb_dir"  
}

####################################################################################################
# 4. functions of compre lite result with fluid result.
####################################################################################################

function compre_with_fluid {
  cd ${WORKSPACE}/tools/ci_tools
  result=`python3 ci_model_unit_test.py cmp_diff`
}

####################################################################################################
# 5. main function
#    $1: adb device index, eg. 0 1
#    $2: adb workspace directory name
####################################################################################################

function main {
  prepare_models
  compile_according_lib_and_opt
  test_model $1 $2
  compre_with_fluid
}

main $1 $2
