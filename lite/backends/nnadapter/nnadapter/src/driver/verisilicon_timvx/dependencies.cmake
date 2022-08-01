# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
# http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

if(NOT NNADAPTER_VERISILICON_TIMVX_VIV_SDK_ROOT)
  set(NNADAPTER_VERISILICON_TIMVX_VIV_SDK_ROOT $ENV{NNADAPTER_VERISILICON_TIMVX_VIV_SDK_ROOT})
endif()

if(NOT NNADAPTER_VERISILICON_TIMVX_VIV_SDK_URL)
  set(NNADAPTER_VERISILICON_TIMVX_VIV_SDK_URL $ENV{NNADAPTER_VERISILICON_TIMVX_VIV_SDK_URL})
endif()

include(ExternalProject)

set(VERISILICON_TIMVX_PROJECT extern_tim-vx)
set(VERISILICON_TIMVX_SOURCES_DIR ${THIRD_PARTY_PATH}/tim-vx)
set(VERISILICON_TIMVX_INSTALL_DIR ${THIRD_PARTY_PATH}/install/tim-vx)
set(VERISILICON_TIMVX_BUILD_COMMAND $(MAKE) -j)

if(NOT NNADAPTER_VERISILICON_TIMVX_VIV_SDK_ROOT)
  if(NOT NNADAPTER_VERISILICON_TIMVX_VIV_SDK_URL)
    message(FATAL_ERROR "Must set NNADAPTER_VERISILICON_TIMVX_VIV_SDK_URL or env NNADAPTER_VERISILICON_TIMVX_VIV_SDK_URL when NNADAPTER_WITH_VERISILICON_TIMVX=ON")
  endif()
  set(VERISILICON_TIMVX_PREBUILT_SDK_DIR ${VERISILICON_TIMVX_INSTALL_DIR}/prebuilt-sdk)
  file(MAKE_DIRECTORY ${VERISILICON_TIMVX_PREBUILT_SDK_DIR})
  get_filename_component(VERISILICON_TIMVX_VIV_SDK_FILE_NAME ${NNADAPTER_VERISILICON_TIMVX_VIV_SDK_URL} NAME_WE)
  get_filename_component(VERISILICON_TIMVX_VIV_SDK_FILE_EXT ${NNADAPTER_VERISILICON_TIMVX_VIV_SDK_URL} EXT)
  set(VERISILICON_TIMVX_VIV_SDK_FILE_PATH ${VERISILICON_TIMVX_PREBUILT_SDK_DIR}/${VERISILICON_TIMVX_VIV_SDK_FILE_NAME}${VERISILICON_TIMVX_VIV_SDK_FILE_EXT})
  message(STATUS "Downloading and extract external VIV SDK from ${NNADAPTER_VERISILICON_TIMVX_VIV_SDK_URL} to ${VERISILICON_TIMVX_PREBUILT_SDK_DIR} ...")
  file(DOWNLOAD ${NNADAPTER_VERISILICON_TIMVX_VIV_SDK_URL}
    ${VERISILICON_TIMVX_VIV_SDK_FILE_PATH}
    SHOW_PROGRESS)
  execute_process(COMMAND
    tar -xf ${VERISILICON_TIMVX_VIV_SDK_FILE_PATH} -C ${VERISILICON_TIMVX_PREBUILT_SDK_DIR}/)
  set(NNADAPTER_VERISILICON_TIMVX_VIV_SDK_ROOT ${VERISILICON_TIMVX_PREBUILT_SDK_DIR}/${VERISILICON_TIMVX_VIV_SDK_FILE_NAME})
endif()
message(STATUS "NNADAPTER_VERISILICON_TIMVX_VIV_SDK_ROOT=${NNADAPTER_VERISILICON_TIMVX_VIV_SDK_ROOT}")

# Remove the -Werror flags to avoid compilation errors 
set(VERISILICON_TIMVX_PATCH_COMMAND sed -e "s/-Werror//g" -i CMakeLists.txt && sed -e "s/3.14/3.10/g" -i CMakeLists.txt)
if(CMAKE_SYSTEM_NAME MATCHES "Android")
  # Hack the TIM-VX and change the name of lib 'libArchModelSw.so' to 'libarchmodelSw.so' for Android
  set(VERISILICON_TIMVX_PATCH_COMMAND ${VERISILICON_TIMVX_PATCH_COMMAND} && sed -e "s/libArchModelSw/libarchmodelSw/g" -i cmake/local_sdk.cmake)
  # Fix the compilation error: "src/tim/vx/ops/custom_base.cc:165:44: error: variable-sized object may not be initialized"
  # Don't use ";" in command string but use $<SEMICOLON> instead, refer to https://stackoverflow.com/questions/43398478/how-to-print-a-symbol-using-cmake-command for more details
  # set(VERISILICON_TIMVX_PATCH_COMMAND ${VERISILICON_TIMVX_PATCH_COMMAND} && sed -e "s/vsi_nn_kernel_node_param_t node_params\\[param_num\\] = {NULL}$<SEMICOLON>/vsi_nn_kernel_node_param_t node_params[param_num]$<SEMICOLON> memset(node_params, 0, sizeof(vsi_nn_kernel_node_param_t) * param_num)$<SEMICOLON>/g" -i src/tim/vx/ops/custom_base.cc)
endif()

ExternalProject_Add(
  ${VERISILICON_TIMVX_PROJECT}
  ${EXTERNAL_PROJECT_LOG_ARGS}
  GIT_REPOSITORY      "https://github.com/VeriSilicon/TIM-VX.git"
  GIT_TAG             ${NNADAPTER_VERISILICON_TIMVX_SRC_GIT_TAG}
  GIT_CONFIG          user.name=anonymous user.email=anonymous@anonymous.com
  SOURCE_DIR          ${VERISILICON_TIMVX_SOURCES_DIR}
  PREFIX              ${VERISILICON_TIMVX_INSTALL_DIR}
  PATCH_COMMAND       ${VERISILICON_TIMVX_PATCH_COMMAND}
  BUILD_COMMAND       ${VERISILICON_TIMVX_BUILD_COMMAND}
  CMAKE_ARGS          -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
                      -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
                      -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
                      -DEXTERNAL_VIV_SDK=${NNADAPTER_VERISILICON_TIMVX_VIV_SDK_ROOT}
                      -DCMAKE_INSTALL_PREFIX=${VERISILICON_TIMVX_INSTALL_DIR}
                      ${CROSS_COMPILE_CMAKE_ARGS}
)

set(VERISILICON_TIMVX_LIBRARY "${VERISILICON_TIMVX_INSTALL_DIR}/lib/libtim-vx.so")

add_custom_target(copy_timvx_libs
  COMMAND cp -r ${VERISILICON_TIMVX_LIBRARY} ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${VERISILICON_TIMVX_PROJECT}
)

add_library(timvx_libs SHARED IMPORTED GLOBAL)
set_property(TARGET timvx_libs PROPERTY IMPORTED_LOCATION ${VERISILICON_TIMVX_LIBRARY})
add_dependencies(timvx_libs copy_timvx_libs)

include_directories(${VERISILICON_TIMVX_INSTALL_DIR}/include)

set(DEPS ${DEPS} timvx_libs)
