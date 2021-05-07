#!/bin/bash
set -e
set +x
set -e
#####################################################################################################
# 1. global variables, you can change them according to your requirements
#####################################################################################################
# num of threads used during compiling..
readonly NUM_PROC=${LITE_BUILD_THREADS:-4}
#####################################################################################################




#####################################################################################################
# 2. local variables, these variables should not be changed.
#####################################################################################################
# absolute path of Paddle-Lite.
readonly workspace=$PWD/$(dirname $0)/../../
# on mac environment, we should expand the maximum file num to compile successfully
os_name=`uname -s`
if [ ${os_name} == "Darwin" ]; then
   ulimit -n 1024
fi
#####################################################################################################



####################################################################################################
# 4. compiling functions
# 4.1 function of tiny_publish compiling
# here we only compile light_api lib
function make_fbs {
  build_dir=$workspace/build.lite.flatbuffer
  if [ -d $build_dir ]
  then
      rm -rf $build_dir
  fi
  mkdir -p $build_dir
  cd $build_dir

  cmake $workspace   \
      -DWITH_LITE=ON \
      -DLITE_WITH_ARM=ON \
      -DLITE_WITH_X86=OFF \
      -DWITH_TESTING=OFF \
      -DLITE_WITH_LIGHT_WEIGHT_FRAMEWORK=ON \
      -DARM_TARGET_OS=android \
      -DLITE_UPDATE_FBS_HEAD=ON \
      -DLITE_ON_TINY_PUBLISH=ON 

  make fbs_headers -j$NUM_PROC

  # clean fbs-pre-build
  git checkout $workspace/third-party/flatbuffers/pre-build/ && rm -rf $workspace/third-party/flatbuffers/pre-build/*
  # update
  cp -rf $workspace/lite/model_parser/flatbuffers/*generated.h $workspace/third-party/flatbuffers/pre-build/
  cp -rf $workspace/lite/backends/opencl/utils/*generated.h $workspace/third-party/flatbuffers/pre-build/
  cp -rf third_party/install/flatbuffers/include/flatbuffers $workspace/third-party/flatbuffers/pre-build/
  # return to original path
  cd -
}

make_fbs
echo "Success! Flatbuffer module has been updated into: $workspace/third-party/flatbuffers/pre-build"
