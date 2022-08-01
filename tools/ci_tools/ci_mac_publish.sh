#!/bin/bash
#
# Start the CI task for inference lib compiling on MacOS platform.
set +x
set -e

#####################################################################################################
# 1. global variables, you can change them according to your requirements
#####################################################################################################
# Python version
PYTHON_VERSION=(3.7)
# Absolute path of Paddle-Lite source code.
SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
WORKSPACE=${SHELL_FOLDER%tools/ci_tools*}
# OpenCL
BUILD_OPENCL=ON
# Common options
BUILD_EXTRA=ON
WITH_EXCEPTION=ON
WITH_PROFILE=ON
WITH_PRECISION_PROFILE=ON

# Model download url
mobilenet_v1_url=http://paddle-inference-dist.bj.bcebos.com/mobilenet_v1.tar.gz

####################################################################################################
# Functions of downloading model file
# Arguments:
#   1. model_name
#   2. downloading url
####################################################################################################
function prepare_model {
  local model_name=$1
  local download_url=$2
  wget $download_url
  tar zxf $model_name.tar.gz
}

####################################################################################################
# Functions of MacOS compiling test.
# Globals:
#   WORKSPACE
# Arguments:
#   1. python version
####################################################################################################
function publish_inference_lib {
  cd $WORKSPACE
  # Local variables
  python_version=$1
  # Remove Compiling Cache
  rm -rf build*
  for python_version in ${PYTHON_VERSION[@]}; do
    # Step1. Compiling python installer on mac
    ./lite/tools/build.sh \
      --build_python=ON \
      --python_version=$python_version \
      --build_opencl=$BUILD_OPENCL \
      --build_extra=$BUILD_EXTRA \
      --with_exception=$WITH_EXCEPTION \
      --with_profile=$WITH_PROFILE \
      --with_precision_profile=$WITH_PRECISION_PROFILE \
      x86
    # Step2. Checking results: cplus and python inference lib.
    build_dir=build.lite.x86
    if [ ${BUILD_OPENCL} = ON ]; then
      build_dir=build.lite.x86.opencl
    fi

    if [ -d ${build_dir}/inference_lite_lib/python/install/dist ]; then
#      macOS python installer is not supported because pybind is compatible to MacOs of high version
#      # test python installer
      cd ${build_dir}
#      cd inference_lite_lib/python/install/dist/
#
#      # Here is a temporary solution for pip bug on macOS,
#      # When compile a python module installer on mac whose system is higher than 11,
#      # the resulted installer can not be installed on current machine.
#      installer_name=$(ls)
#      macOS10_installer_name=$(ls | sed 's/11/10/g')
#      mv $installer_name $macOS10_installer_name && cd -
#      python$python_version -m pip install --force-reinstall  inference_lite_lib/python/install/dist/*.whl
      # download test model
      prepare_model mobilenet_v1 $mobilenet_v1_url
#      # test opt
#      paddle_lite_opt
#      paddle_lite_opt --model_dir=mobilenet_v1 --optimize_out=mobilenet_v1_arm
#      paddle_lite_opt --model_dir=mobilenet_v1 --valid_targets=x86 --optimize_out=mobilenet_v1_x86
#      paddle_lite_opt --model_dir=mobilenet_v1 --valid_targets=x86_opencl --optimize_out=mobilenet_v1_x86_opencl
#      # test inference demo
#      cd inference_lite_lib/demo/python
#      python$python_version mobilenetv1_full_api.py  --model_dir=$WORKSPACE/${build_dir}/mobilenet_v1
#      python$python_version mobilenetv1_light_api.py  --model_dir=$WORKSPACE/${build_dir}/mobilenet_v1_x86.nb
#      python$python_version mobilenetv1_light_api.py  --model_dir=$WORKSPACE/${build_dir}/mobilenet_v1_x86_opencl.nb
#      # uninstall
#      python$python_version -m pip uninstall -y paddlelite
      echo "MacOs python installer is ready."
    else
      # Error message.
      echo "**************************************************************************************"
      echo -e "* Mac python installer compiling task failed on the following instruction:"
      echo -e "*     ./lite/tools/build.sh --with_python=ON --python_version=$python_version
      --build_opencl=$BUILD_OPENCL --build_extra=$BUILD_EXTRA --with_exception=$WITH_EXCEPTION
      --with_profile=$WITH_PROFILE --with_precision_profile=$WITH_PRECISION_PROFILE x86"
      echo "**************************************************************************************"
      exit 1
    fi

    # Test x86 cxx demo
    local cxx_demo_dir=${WORKSPACE}/${build_dir}/inference_lite_lib/demo/cxx/
    if [ -d ${cxx_demo_dir} ]; then
      # full demo
      cd ${cxx_demo_dir}/mobilenetv1_full/
      sh build.sh
      ./mobilenet_full_api $WORKSPACE/${build_dir}/mobilenet_v1  1,3,224,224  10  2  0

      # light demo
#      cd ${cxx_demo_dir}/mobilenetv1_light/
#      sh build.sh
#      ./mobilenet_light_api $WORKSPACE/${build_dir}/mobilenet_v1_x86_opencl.nb 1,3,224,224  10  2  0
    else
      echo -e "Directory: ${cxx_demo_dir} not found!"
      exit 1
    fi
  done
}

function publish_metal_lib {
  cd $WORKSPACE
  # Local variables
  python_version=$1
  with_metal=$2
  with_opencl=$3
  # Remove Compiling Cache
  rm -rf build*

  # Step1. Compiling python installer on mac
  ./lite/tools/build_linux.sh \
    --with_python=ON \
    --python_version=$python_version \
    --with_opencl=${with_opencl} \
    --with_metal=${with_metal} \
    --with_extra=$BUILD_EXTRA \
    --with_exception=$WITH_EXCEPTION \
    --with_profile=$WITH_PROFILE \
    --with_precision_profile=$WITH_PRECISION_PROFILE \
    --arch=x86

  # Step2. Checking results: cplus and python inference lib.
  build_dir=build.lite.linux.x86.gcc
  if [ ${with_opencl} == ON ]; then
    build_dir=${build_dir}.opencl
  fi
  if [ ${with_metal} == ON ]; then
    build_dir=${build_dir}.metal
  fi

  if [ -d ${build_dir}/inference_lite_lib/python/install/dist ]; then
    cd ${build_dir}
    prepare_model mobilenet_v1 $mobilenet_v1_url
  else
    echo "dist not found."
    exit 1
  fi

  # Step3. Test x86 cxx demo
  local cxx_demo_dir=${WORKSPACE}/${build_dir}/inference_lite_lib/demo/cxx/
  if [ -d ${cxx_demo_dir} ]; then
    # full demo
    cd ${cxx_demo_dir}/mobilenetv1_full/
    sh build.sh --with_metal=${with_metal}
    # ./mobilenet_full_api $WORKSPACE/${build_dir}/mobilenet_v1  1,3,224,224  10  2  0

    # light demo
    # cd ${cxx_demo_dir}/mobilenetv1_light/
    # sh build.sh
    # ./mobilenet_light_api $WORKSPACE/${build_dir}/mobilenet_v1_x86_opencl.nb 1,3,224,224  10  2  0
  else
    echo -e "Directory: ${cxx_demo_dir} not found!"
    exit 1
  fi
}

# Compiling test
for version in ${PYTHON_VERSION[@]}; do
    publish_inference_lib $version
done

# Compiling test (use Metal)
# publish_metal_lib ${py_version} ${with_metal} ${with_opencl}
for version in ${PYTHON_VERSION[@]}; do
    publish_metal_lib $version ON OFF
done
