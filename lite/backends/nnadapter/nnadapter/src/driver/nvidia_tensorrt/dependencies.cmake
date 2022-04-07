# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

if(NOT DEFINED NNADAPTER_NVIDIA_CUDA_ROOT)
  set(NNADAPTER_NVIDIA_CUDA_ROOT $ENV{NNADAPTER_NVIDIA_CUDA_ROOT})
  if(NOT NNADAPTER_NVIDIA_CUDA_ROOT)
    message(FATAL_ERROR "Must set NNADAPTER_NVIDIA_CUDA_ROOT or env NNADAPTER_NVIDIA_CUDA_ROOT when NNADAPTER_WITH_NVIDIA_TENSORRT=ON")
  endif()
endif()
message(STATUS "NNADAPTER_NVIDIA_CUDA_ROOT: ${NNADAPTER_NVIDIA_CUDA_ROOT}")

if(NOT DEFINED NNADAPTER_NVIDIA_TENSORRT_ROOT)
  set(NNADAPTER_NVIDIA_TENSORRT_ROOT $ENV{NNADAPTER_NVIDIA_TENSORRT_ROOT})
  if(NOT NNADAPTER_NVIDIA_TENSORRT_ROOT)
    message(FATAL_ERROR "Must set NNADAPTER_NVIDIA_TENSORRT_ROOT or env NNADAPTER_NVIDIA_TENSORRT_ROOT when NNADAPTER_WITH_NVIDIA_TENSORRT=ON")
  endif()
endif()
message(STATUS "NNADAPTER_NVIDIA_TENSORRT_ROOT: ${NNADAPTER_NVIDIA_TENSORRT_ROOT}")

if(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64" OR CMAKE_SYSTEM_PROCESSOR STREQUAL "amd64")
else()
  message(FATAL_ERROR "${CMAKE_SYSTEM_PROCESSOR} isn't supported by NVIDIA GPU.")
endif()

# find cuda/cudnn/tensorrt include dirs
find_path(NNADAPTER_NVIDIA_CUDA_INCLUDE_DIR cuda.h
          PATHS ${NNADAPTER_NVIDIA_CUDA_ROOT}
                ${NNADAPTER_NVIDIA_CUDA_ROOT}/include
                /usr/include/${CMAKE_SYSTEM_PROCESSOR}-linux-gnu
                /usr/include
          NO_DEFAULT_PATH
)
find_path(NNADAPTER_NVIDIA_CUDNN_INCLUDE_DIR cudnn.h
          PATHS ${NNADAPTER_NVIDIA_CUDA_ROOT}
                ${NNADAPTER_NVIDIA_CUDA_ROOT}/include
                /usr/include/${CMAKE_SYSTEM_PROCESSOR}-linux-gnu
                /usr/include
          NO_DEFAULT_PATH
)
find_path(NNADAPTER_NVIDIA_TENSORRT_INCLUDE_DIR NvInfer.h
          PATHS ${NNADAPTER_NVIDIA_TENSORRT_ROOT}
                ${NNADAPTER_NVIDIA_TENSORRT_ROOT}/include
                /usr/include/${CMAKE_SYSTEM_PROCESSOR}-linux-gnu
                /usr/include
          NO_DEFAULT_PATH
)
include_directories(${NNADAPTER_NVIDIA_CUDA_INCLUDE_DIR}
                    ${NNADAPTER_NVIDIA_CUDNN_INCLUDE_DIR}
                    ${NNADAPTER_NVIDIA_TENSORRT_INCLUDE_DIR})


# find NvInferVersion.h and get version info
file(READ ${NNADAPTER_NVIDIA_TENSORRT_INCLUDE_DIR}/NvInferVersion.h TENSORRT_VERSION_FILE_CONTENTS)
string(REGEX MATCH "define NV_TENSORRT_MAJOR +([0-9]+)" TENSORRT_MAJOR_VERSION "${TENSORRT_VERSION_FILE_CONTENTS}")
string(REGEX MATCH "define NV_TENSORRT_MINOR +([0-9]+)" TENSORRT_MINOR_VERSION "${TENSORRT_VERSION_FILE_CONTENTS}")
string(REGEX MATCH "define NV_TENSORRT_PATCH +([0-9]+)" TENSORRT_PATCH_VERSION "${TENSORRT_VERSION_FILE_CONTENTS}")
string(REGEX MATCH "define NV_TENSORRT_BUILD +([0-9]+)" TENSORRT_BUILD_VERSION "${TENSORRT_VERSION_FILE_CONTENTS}")
if("${TENSORRT_MAJOR_VERSION}" STREQUAL "")
  message(FATAL_ERROR "Failed to detect TensorRT version.")
endif()
string(REGEX REPLACE "define NV_TENSORRT_MAJOR +([0-9]+)" "\\1" TENSORRT_MAJOR_VERSION "${TENSORRT_MAJOR_VERSION}")
string(REGEX REPLACE "define NV_TENSORRT_MINOR +([0-9]+)" "\\1" TENSORRT_MINOR_VERSION "${TENSORRT_MINOR_VERSION}")
string(REGEX REPLACE "define NV_TENSORRT_PATCH +([0-9]+)" "\\1" TENSORRT_PATCH_VERSION "${TENSORRT_PATCH_VERSION}")
string(REGEX REPLACE "define NV_TENSORRT_BUILD +([0-9]+)" "\\1" TENSORRT_BUILD_VERSION "${TENSORRT_BUILD_VERSION}")
message(STATUS "Detected tensorrt version: ${TENSORRT_MAJOR_VERSION}.${TENSORRT_MINOR_VERSION}.${TENSORRT_PATCH_VERSION}.${TENSORRT_BUILD_VERSION}")
add_compile_options(-DTENSORRT_MAJOR_VERSION=${TENSORRT_MAJOR_VERSION})
add_compile_options(-DTENSORRT_MINOR_VERSION=${TENSORRT_MINOR_VERSION})
add_compile_options(-DTENSORRT_PATCH_VERSION=${TENSORRT_PATCH_VERSION})
add_compile_options(-DTENSORRT_BUILD_VERSION=${TENSORRT_BUILD_VERSION})

# find cuda/cudnn/tensorrt lib dirs
find_path(NNADAPTER_NVIDIA_CUDA_LIB_DIR libcudart.so
          PATHS ${NNADAPTER_NVIDIA_CUDA_ROOT}
                ${NNADAPTER_NVIDIA_CUDA_ROOT}/lib64
                ${NNADAPTER_NVIDIA_CUDA_ROOT}/lib
                ${NNADAPTER_NVIDIA_CUDA_ROOT}/targets/${NNADAPTER_NVIDIA_TENSORRT_ROOT}-linux/lib
                /usr/lib/${CMAKE_SYSTEM_PROCESSOR}-linux-gnu
                /usr/lib64
                /usr/lib
          NO_DEFAULT_PATH
)
find_path(NNADAPTER_NVIDIA_CUDNN_LIB_DIR libcudnn.so
          PATHS ${NNADAPTER_NVIDIA_CUDA_ROOT}
                ${NNADAPTER_NVIDIA_CUDA_ROOT}/lib64
                ${NNADAPTER_NVIDIA_CUDA_ROOT}/lib
                ${NNADAPTER_NVIDIA_CUDA_ROOT}/targets/${NNADAPTER_NVIDIA_TENSORRT_ROOT}-linux/lib
                /usr/lib/${CMAKE_SYSTEM_PROCESSOR}-linux-gnu
                /usr/lib64
                /usr/lib
          NO_DEFAULT_PATH
)
find_path(NNADAPTER_NVIDIA_TENSORRT_LIB_DIR libnvinfer.so
          PATHS ${NNADAPTER_NVIDIA_TENSORRT_ROOT}
                ${NNADAPTER_NVIDIA_TENSORRT_ROOT}/lib64
                ${NNADAPTER_NVIDIA_TENSORRT_ROOT}/lib
                /usr/lib/${CMAKE_SYSTEM_PROCESSOR}-linux-gnu
                /usr/lib64
                /usr/lib
          NO_DEFAULT_PATH
)

# CUDA libraries
# cudart.so 
find_library(NVIDIA_CUDA_CUDART_FILE NAMES cudart
  PATHS ${NNADAPTER_NVIDIA_CUDA_LIB_DIR}
  CMAKE_FIND_ROOT_PATH_BOTH)
if(NOT NVIDIA_CUDA_CUDART_FILE)
  message(FATAL_ERROR "Missing libcudart.so in ${NNADAPTER_NVIDIA_CUDA_LIB_DIR}")
endif()
add_library(cuda_cudart SHARED IMPORTED GLOBAL)
set_property(TARGET cuda_cudart PROPERTY IMPORTED_LOCATION ${NVIDIA_CUDA_CUDART_FILE})

# cudnn.so 
find_library(NVIDIA_CUDA_CUDNN_FILE NAMES cudnn
  PATHS ${NNADAPTER_NVIDIA_CUDNN_LIB_DIR}
  CMAKE_FIND_ROOT_PATH_BOTH)
if(NOT NVIDIA_CUDA_CUDNN_FILE)
  message(FATAL_ERROR "Missing libcudnn.so in ${NNADAPTER_NVIDIA_CUDNN_LIB_DIR}")
endif()
add_library(cuda_cudnn SHARED IMPORTED GLOBAL)
set_property(TARGET cuda_cudnn PROPERTY IMPORTED_LOCATION ${NVIDIA_CUDA_CUDNN_FILE})

# TENSORRT libraries
# libnvinfer.so
find_library(NVIDIA_TENSORRT_NVINFER_FILE NAMES nvinfer
  PATHS ${NNADAPTER_NVIDIA_TENSORRT_LIB_DIR}
  CMAKE_FIND_ROOT_PATH_BOTH)
if(NOT NVIDIA_TENSORRT_NVINFER_FILE)
  message(FATAL_ERROR "Missing libnvinfer.so in ${NNADAPTER_NVIDIA_TENSORRT_LIB_DIR}")
endif()
add_library(tensorrt_nvinfer SHARED IMPORTED GLOBAL)
set_property(TARGET tensorrt_nvinfer PROPERTY IMPORTED_LOCATION ${NVIDIA_TENSORRT_NVINFER_FILE})

# libnvinfer_plugin.so
find_library(NVIDIA_TENSORRT_NVINFER_PLUGIN_FILE NAMES nvinfer_plugin
  PATHS ${NNADAPTER_NVIDIA_TENSORRT_LIB_DIR}
  CMAKE_FIND_ROOT_PATH_BOTH)
if(NOT NVIDIA_TENSORRT_NVINFER_PLUGIN_FILE)
  message(FATAL_ERROR "Missing libnvinfer_plugin.so in ${NNADAPTER_NVIDIA_TENSORRT_LIB_DIR}")
endif()
add_library(tensorrt_nvinfer_plugin SHARED IMPORTED GLOBAL)
set_property(TARGET tensorrt_nvinfer_plugin PROPERTY IMPORTED_LOCATION ${NVIDIA_TENSORRT_NVINFER_PLUGIN_FILE})

set(DEPS ${DEPS} tensorrt_nvinfer tensorrt_nvinfer_plugin cuda_cudnn cuda_cudart)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=deprecated-declarations -Wno-deprecated-declarations")
