# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.
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

if(NOT WITH_PYTHON)
    add_definitions(-DPADDLE_NO_PYTHON)
endif(NOT WITH_PYTHON)

if(WITH_DSO)
    add_definitions(-DPADDLE_USE_DSO)
endif(WITH_DSO)

if(WITH_TESTING)
    add_definitions(-DPADDLE_WITH_TESTING)
endif(WITH_TESTING)

if(NOT WITH_PROFILER)
    add_definitions(-DPADDLE_DISABLE_PROFILER)
endif(NOT WITH_PROFILER)

if(WITH_AVX AND AVX_FOUND)
    set(SIMD_FLAG ${AVX_FLAG})
elseif(SSE3_FOUND)
    set(SIMD_FLAG ${SSE3_FLAG})
endif()

if(WIN32)
  # windows header option for all targets.
  add_definitions(-D_XKEYCHECK_H)
  add_definitions(/openmp)

  if (NOT MSVC)
    message(FATAL "Windows build only support msvc. Which was binded by the nvcc compiler of NVIDIA.")
  endif(NOT MSVC)

  # set the generator platform for ninja
  if(${CMAKE_GENERATOR} MATCHES "Ninja")
      if (NOT ARCH)
          set(ARCH i386)
      endif()

      if (NOT DEFINED CMAKE_GENERATOR_PLATFORM)
          if(ARCH STREQUAL "amd64")
              set(CMAKE_GENERATOR_PLATFORM "x64")
          elseif(ARCH STREQUAL "arm")
              set(CMAKE_GENERATOR_PLATFORM "ARM")
          elseif(ARCH STREQUAL "arm64")
              set(CMAKE_GENERATOR_PLATFORM "ARM64")
          else()
              set(CMAKE_GENERATOR_PLATFORM "Win32")
          endif()
      endif()
  endif()

endif(WIN32)

if(LITE_WITH_CUDA)
    add_definitions(-DLITE_WITH_CUDA)
    add_definitions(-DEIGEN_USE_GPU)

    FIND_PACKAGE(CUDA REQUIRED)

    if(${CUDA_VERSION_MAJOR} VERSION_LESS 7)
        message(FATAL_ERROR "Paddle needs CUDA >= 7.0 to compile")
    endif()

    if(NOT CUDNN_FOUND)
        message(FATAL_ERROR "Paddle needs cudnn to compile")
    endif()
    if(CUPTI_FOUND)
        include_directories(${CUPTI_INCLUDE_DIR})
        add_definitions(-DPADDLE_WITH_CUPTI)
    else()
        message(STATUS "Cannot find CUPTI, GPU Profiling is incorrect.")
    endif()
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-Xcompiler ${SIMD_FLAG}")

    # Include cuda and cudnn
    include_directories(${CUDNN_INCLUDE_DIR})
    include_directories(${CUDA_TOOLKIT_INCLUDE})

elseif(WITH_AMD_GPU)
    add_definitions(-DPADDLE_WITH_HIP)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -D__HIP_PLATFORM_HCC__")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__HIP_PLATFORM_HCC__")
else()
    add_definitions(-DHPPL_STUB_FUNC)
    list(APPEND CMAKE_CXX_SOURCE_FILE_EXTENSIONS cu)
endif()

if (WITH_MKLML AND MKLML_IOMP_LIB)
    message(STATUS "Enable Intel OpenMP with ${MKLML_IOMP_LIB}")
    if(WIN32 OR APPLE)
        # openmp not support well for now on windows
        set(OPENMP_FLAGS "")
    else(WIN32)
        set(OPENMP_FLAGS "-fopenmp")
    endif(WIN32 OR APPLE)

    set(CMAKE_C_CREATE_SHARED_LIBRARY_FORBIDDEN_FLAGS ${OPENMP_FLAGS})
    set(CMAKE_CXX_CREATE_SHARED_LIBRARY_FORBIDDEN_FLAGS ${OPENMP_FLAGS})
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OPENMP_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OPENMP_FLAGS}")
    #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
endif()

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${SIMD_FLAG}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SIMD_FLAG}")

if(WITH_DISTRIBUTE)
  add_definitions(-DPADDLE_WITH_DISTRIBUTE)
endif()

if(WITH_GRPC)
    add_definitions(-DPADDLE_WITH_GRPC)
endif(WITH_GRPC)

if(WITH_BRPC_RDMA)
    add_definitions(-DPADDLE_WITH_BRPC_RDMA)
endif(WITH_BRPC_RDMA)

if(ON_INFER)
    add_definitions(-DPADDLE_ON_INFERENCE)
endif(ON_INFER)

if(WITH_WBAES)
    add_definitions(-DPADDLE_WITH_WBAES)
endif(WITH_WBAES)

if (REPLACE_ENFORCE_GLOG)
  add_definitions("-DREPLACE_ENFORCE_GLOG")
endif()

# for lite
# TODO(Superjomn) not work fine with the option
if (LITE_WITH_X86)
    add_definitions("-DLITE_WITH_X86")
endif()

if (LITE_WITH_ARM)
    add_definitions("-DLITE_WITH_ARM")
endif()

if(LITE_WITH_ARM82_FP16)
    add_definitions(-DENABLE_ARM_FP16)
endif()

if (LITE_WITH_CV)
    if(NOT LITE_WITH_ARM)
        message(FATAL_ERROR "CV functions uses the ARM instructions, so LITE_WITH_ARM must be turned on")
    endif()
    add_definitions("-DLITE_WITH_CV")
endif()

if (LITE_WITH_TRAIN)
    add_definitions("-DLITE_WITH_TRAIN")
endif()

if (WITH_ARM_DOTPROD)
    add_definitions("-DWITH_ARM_DOTPROD")
endif()

if (LITE_WITH_NPU)
    add_definitions("-DLITE_WITH_NPU")
endif()

if (LITE_WITH_XPU)
    add_definitions("-DLITE_WITH_XPU")
endif()

if (LITE_WITH_OPENCL)
    add_definitions("-DLITE_WITH_OPENCL")
endif()

if (LITE_WITH_METAL)
    find_library(METAL_LIBRARY Metal REQUIRED)
    find_library(GRAPHIC CoreGraphics REQUIRED)
    find_library(MPS_LIBRARY MetalPerformanceShaders REQUIRED)
    find_library(FOUNDATION_LIBRARY Foundation)
    add_definitions("-DLITE_WITH_METAL")
endif()


if (LITE_WITH_FPGA)
add_definitions("-DLITE_WITH_FPGA")
endif()

if (LITE_WITH_INTEL_FPGA)
add_definitions("-DLITE_WITH_INTEL_FPGA")
endif()

if (LITE_WITH_BM)
add_definitions("-DLITE_WITH_BM")
endif()

if (LITE_WITH_MLU)
add_definitions("-DLITE_WITH_MLU")
endif()

if (LITE_WITH_NNADAPTER)
  add_definitions(-DLITE_WITH_NNADAPTER)
  # The following macros are only for unit test
  if (WITH_TESTING)
    if (NNADAPTER_WITH_ROCKCHIP_NPU)
      add_definitions("-DNNADAPTER_WITH_ROCKCHIP_NPU")
    endif()
    if (NNADAPTER_WITH_IMAGINATION_NNA)
      add_definitions("-DNNADAPTER_WITH_IMAGINATION_NNA")
    endif()
    if (NNADAPTER_WITH_HUAWEI_ASCEND_NPU)
      add_definitions("-DNNADAPTER_WITH_HUAWEI_ASCEND_NPU")
    endif()
    if (NNADAPTER_WITH_HUAWEI_KIRIN_NPU)
      add_definitions("-DNNADAPTER_WITH_HUAWEI_KIRIN_NPU")
    endif()
    if (NNADAPTER_WITH_MEDIATEK_APU)
      add_definitions("-DNNADAPTER_WITH_MEDIATEK_APU")
    endif()
    if (NNADAPTER_WITH_AMLOGIC_NPU)
      add_definitions("-DNNADAPTER_WITH_AMLOGIC_NPU")
    endif()
    if (NNADAPTER_WITH_CAMBRICON_MLU)
      add_definitions("-DNNADAPTER_WITH_CAMBRICON_MLU")
    endif()
    if (NNADAPTER_WITH_VERISILICON_TIMVX)
      add_definitions("-DNNADAPTER_WITH_VERISILICON_TIMVX")
    endif()
    if (NNADAPTER_WITH_NVIDIA_TENSORRT)
      add_definitions("-DNNADAPTER_WITH_NVIDIA_TENSORRT")
    endif()
    if (NNADAPTER_WITH_QUALCOMM_QNN)
      add_definitions("-DNNADAPTER_WITH_QUALCOMM_QNN")
    endif()
    if (NNADAPTER_WITH_KUNLUNXIN_XTCL)
      add_definitions("-DNNADAPTER_WITH_KUNLUNXIN_XTCL")
    endif()
    if (NNADAPTER_WITH_ANDROID_NNAPI)
      add_definitions("-DNNADAPTER_WITH_ANDROID_NNAPI")
    endif()
    if (NNADAPTER_WITH_INTEL_OPENVINO)
      add_definitions("-DNNADAPTER_WITH_INTEL_OPENVINO")
    endif()
    if (NNADAPTER_WITH_GOOGLE_XNNPACK)
      add_definitions("-DNNADAPTER_WITH_GOOGLE_XNNPACK")
    endif()
  endif()
endif()

if (LITE_WITH_PROFILE)
    add_definitions("-DLITE_WITH_PROFILE")
endif()

if (LITE_WITH_XCODE)
    add_definitions("-DLITE_WITH_XCODE")
endif()

if (LITE_WITH_PRECISION_PROFILE)
    add_definitions("-DLITE_WITH_PRECISION_PROFILE")
endif()

if (LITE_WITH_LOG)
  add_definitions("-DLITE_WITH_LOG")
endif()

if (LITE_WITH_EXCEPTION)
  add_definitions("-DLITE_WITH_EXCEPTION")
endif()

if (LITE_ON_TINY_PUBLISH)
  add_definitions("-DLITE_ON_TINY_PUBLISH")
  add_definitions("-DLITE_ON_FLATBUFFERS_DESC_VIEW")
  add_definitions("-DLITE_WITH_FLATBUFFERS_DESC")
  message(STATUS "Flatbuffers will be used as cpp default program description.")
else()
  add_definitions("-DLITE_WITH_FLATBUFFERS_DESC")
endif()

if (LITE_ON_MODEL_OPTIMIZE_TOOL)
  add_definitions("-DLITE_ON_MODEL_OPTIMIZE_TOOL")
endif(LITE_ON_MODEL_OPTIMIZE_TOOL)

if (LITE_BUILD_EXTRA)
  add_definitions("-DLITE_BUILD_EXTRA")
endif(LITE_BUILD_EXTRA)

if (LITE_WITH_PYTHON)
  add_definitions("-DLITE_WITH_PYTHON")
endif(LITE_WITH_PYTHON)

if (LITE_WITH_ARM82_FP16)
  add_definitions("-DLITE_WITH_ARM82_FP16")
endif(LITE_WITH_ARM82_FP16)

if (LITE_WITH_ARM8_SVE2)
  add_definitions("-DLITE_WITH_ARM8_SVE2")
endif(LITE_WITH_ARM8_SVE2)

if (LITE_WITH_M1)
add_definitions("-DLITE_WITH_M1")
endif(LITE_WITH_M1)

if (EMSCRIPTEN)
  add_compile_options("-pthread")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pthread")
endif()
