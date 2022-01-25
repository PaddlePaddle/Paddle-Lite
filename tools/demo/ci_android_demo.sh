#!/bin/bash
set -ex

# Absolute path of Paddle-Lite source code.
SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
WORKSPACE=${SHELL_FOLDER%tools/ci_tools*}
NUM_PROC=4

# model name and downloading url

# if operating in mac env, we should expand the maximum file num
os_name=`uname -s`
if [ ${os_name} == "Darwin" ]; then
   ulimit -n 1024
fi

####################################################################################################
# 1. functions of prepare workspace before compiling
####################################################################################################
# 1.1 compile paddlelite android lib
function compile_lib {
  # compile paddlelite lib
  # eg. cd Paddle-lite && ./lite/tools/build_android.sh --with_extra=ON
}

# 1.2 compile_and_install_opt
function compile_and_install_opt {
  # compile_and_install_opt
  # eg. cd Paddle-Lite && ./lite/tools/build_linux.sh --with_python=ON
  #     install paddlelite
  # eg. cd build.lite.x86/inference_lite_lib/python/install/dist && python -m pip install *whl
}

####################################################################################################
# 3. functions of testing demo on android platform.
####################################################################################################

# 3.1 Compile Android demo and upload them onto adb device to perform unit test

function compile_and_test_demo {
  adb_index=$1
  adb_dir=$2

  # download models file
  bash download_models.sh
  # convert model
  paddle_lite_opt ./model --optimize_out=opt_model
  # compile model
  bash compile.sh

  # upload model, lib, bin onto adb devices and run
  adb_devices=($(adb devices |grep -v devices |grep device | awk -F " " '{print $1}'))
   # prepare workspace directory
  adb -s ${adb_devices[$adb_index]} shell "cd /data/local/tmp && rm -rf $adb_dir && mkdir $adb_dir"
  # 1. upload optimized model
  adb -s ${adb_devices[$adb_index]} push opt_model.nb /data/local/tmp/$adb_dir

  # 2. perform armv8 unit_test
  #    3.1 upload armv8 lib
  adb -s ${adb_devices[$adb_index]} push android_lib/armv8.clang/cxx/lib/libpaddle_light_api_shared.so /data/local/tmp/$adb_dir
  #    3.2 compile and upload armv8 demo
  cd android_lib/armv8.clang/demo/cxx/mobile_bin && make && chmod +x mobile_bin
  adb -s ${adb_devices[$adb_index]} push mobile_bin /data/local/tmp/$adb_dir  && cd -

  #    3.3 perform unit test
  adb -s ${adb_devices[$adb_index]} shell "cd /data/local/tmp/$adb_dir && export LD_LIBRARY_PATH=./ &&  ./mobile_bin ./model_opt.nb"
}

function run_test_model {
  # 1. get into demo directory
  cd inference_lite_lib/demo/android
  # 2. compile and test demo
  demos=$(ls)
  for demo_dir in $(demos[@]); do
    cd $demo_dir
    compile_and_test_demo $1 $2
  done
}

####################################################################################################
# 4. main function
#    $1: adb device index, eg. 0 1
#    $2: adb workspace directory name
####################################################################################################

function main {
  # step1. compile paddle-lite android lib
  compile_lib
  # step2. compile and install paddle-lite opt python tool
  compile_and_install_opt
  # step3. compile demo a
  run_test_demo $1 $2
}

main $1 $2
