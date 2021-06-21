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

if (NOT LITE_WITH_XPU)
  return ()
endif ()

INCLUDE (ExternalProject)

set (XPU_SOURCE_DIR ${THIRD_PARTY_PATH}/xpu)
set (XPU_DOWNLOAD_DIR ${XPU_SOURCE_DIR}/download)
set (XPU_INSTALL_DIR ${THIRD_PARTY_PATH}/install)

if (NOT XPU_SDK_URL)
  set (XPU_SDK_URL "https://baidu-kunlun-product.cdn.bcebos.com/KL-SDK/klsdk-dev_paddle")
endif ()

if (NOT XPU_SDK_ENV)
  if (LITE_WITH_X86)
    set (XPU_SDK_ENV "bdcentos_x86_64")
    if (${HOST_SYSTEM} STREQUAL "ubuntu")
      set (XPU_SDK_ENV "ubuntu_x86_64")
    endif ()
  elseif (LITE_WITH_ARM)
    set (XPU_SDK_ENV "kylin_aarch64")
  else ()
    message (FATAL_ERROR "xpu doesn't supported the host env")
  endif ()
endif ()

macro (prepare_xpu_sdk sdk)
  set (xpu_${sdk}_url "${XPU_SDK_URL}/${sdk}-${XPU_SDK_ENV}.tar.gz")
  message (STATUS "xpu_${sdk}_url: ${xpu_${sdk}_url}")

  ExternalProject_Add (
    extern_xpu_${sdk}
    ${EXTERNAL_PROJECT_LOG_ARGS}
    DOWNLOAD_DIR            ${XPU_DOWNLOAD_DIR}
    DOWNLOAD_COMMAND        wget --no-check-certificate -c -q ${xpu_${sdk}_url} && tar xf ${sdk}-${XPU_SDK_ENV}.tar.gz
    CONFIGURE_COMMAND       ""
    BUILD_COMMAND           ""
    UPDATE_COMMAND          ""
    INSTALL_COMMAND         ${CMAKE_COMMAND} -E copy_directory ${XPU_DOWNLOAD_DIR}/${sdk}-${XPU_SDK_ENV} ${XPU_INSTALL_DIR}/xpu/${sdk}
  )

  set (xpu_${sdk}_root        "${XPU_INSTALL_DIR}/xpu/${sdk}"  CACHE PATH "xpu ${sdk} include directory" FORCE)
  set (xpu_${sdk}_include_dir "${xpu_${sdk}_root}/include" CACHE PATH "xpu ${sdk} include directory" FORCE)

  include_directories (${xpu_${sdk}_include_dir})

  foreach (lib ${ARGN})
    add_library (${lib} SHARED IMPORTED GLOBAL)
    set_property (TARGET ${lib} PROPERTY IMPORTED_LOCATION "${xpu_${sdk}_root}/so/lib${lib}.so")
    add_dependencies (${lib} extern_xpu_${sdk})
    link_libraries (${lib})
  endforeach ()
endmacro ()

if (NOT XPU_SDK_ROOT)
  prepare_xpu_sdk (xdnn xpuapi)
  prepare_xpu_sdk (xre xpurt)
  set (xpu_builder_libs xpuapi CACHE INTERNAL "xpu builder libs")
  set (xpu_runtime_libs xpurt CACHE INTERNAL "xpu runtime libs")

  if (LITE_WITH_XTCL)
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DDMLC_USE_GLOG=1")
    prepare_xpu_sdk (xtcl tvm xtcl)
    prepare_xpu_sdk (xtdk xpujitc LLVM-10)
    set (xpu_runtime_libs xpurt LLVM-10 xpujitc CACHE INTERNAL "xpu runtime libs")
    set (xpu_builder_libs xtcl tvm xpuapi CACHE INTERNAL "xpu builder libs")
  endif ()
  return ()
endif ()

# **DEPRECATED**, use XPU_SDK_URL/XPU_SDK_ENV in the future
message (STATUS "XPU_SDK_ROOT: ${XPU_SDK_ROOT}")

set (XPU_XTDK_INCLUDE_DIR    "${XPU_SDK_ROOT}/XTDK/include" CACHE PATH "xpu xtdk include directory" FORCE)
set (XPUAPI_LIB              "${XPU_SDK_ROOT}/XTDK/shlib/libxpuapi.so" CACHE FILEPATH "libxpuapi.so" FORCE)
set (XPURT_LIB               "${XPU_SDK_ROOT}/XTDK/runtime/shlib/libxpurt.so" CACHE FILEPATH "libxpurt.so" FORCE)

include_directories (${XPU_XTDK_INCLUDE_DIR})

add_library (xpuapi SHARED IMPORTED GLOBAL)
set_property (TARGET xpuapi PROPERTY IMPORTED_LOCATION ${XPUAPI_LIB})

add_library (xpurt SHARED IMPORTED GLOBAL)
set_property (TARGET xpurt PROPERTY IMPORTED_LOCATION ${XPURT_LIB})

set (xpu_runtime_libs xpuapi xpurt CACHE INTERNAL "xpu runtime libs")
set (xpu_builder_libs xpuapi xpurt CACHE INTERNAL "xpu builder libs")

if (LITE_WITH_XTCL)
  set (XPU_XTCL_INCLUDE_DIR  "${XPU_SDK_ROOT}/XTCL/include" CACHE PATH "xpu xtcl include directory" FORCE)
  set (XTCL_LIB              "${XPU_SDK_ROOT}/XTCL/shlib/libxtcl.so" CACHE FILEPATH "libxtcl.a" FORCE)
  set (TVM_LIB               "${XPU_SDK_ROOT}/XTCL/shlib/libtvm.so" CACHE FILEPATH "libtvm.so" FORCE)
  set (LLVM_10_LIB           "${XPU_SDK_ROOT}/XTCL/shlib/libLLVM-10.so" CACHE FILEPATH "libLLVM-10.so" FORCE)
  set (XPUJITC_LIB           "${XPU_SDK_ROOT}/XTCL/shlib/libxpujitc.so" CACHE FILEPATH "libxpujitc.so" FORCE)

  include_directories (${XPU_XTCL_INCLUDE_DIR})

  add_library (xtcl SHARED IMPORTED GLOBAL)
  set_property (TARGET xtcl PROPERTY IMPORTED_LOCATION ${XTCL_LIB})

  add_library (tvm SHARED IMPORTED GLOBAL)
  set_property (TARGET tvm PROPERTY IMPORTED_LOCATION ${TVM_LIB})

  add_library (llvm_10 SHARED IMPORTED GLOBAL)
  set_property (TARGET llvm_10 PROPERTY IMPORTED_LOCATION ${LLVM_10_LIB})

  add_library(xpujitc SHARED IMPORTED GLOBAL)
  set_property(TARGET xpujitc PROPERTY IMPORTED_LOCATION ${XPUJITC_LIB})

  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DDMLC_USE_GLOG=1")

  set (xpu_runtime_libs xpurt llvm_10 xpujitc CACHE INTERNAL "xpu runtime libs")
  set (xpu_builder_libs xtcl tvm xpuapi CACHE INTERNAL "xpu builder libs")
endif ()
