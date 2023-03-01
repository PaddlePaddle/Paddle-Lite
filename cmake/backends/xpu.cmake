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
  set (XPU_SDK_URL "https://baidu-kunlun-product.su.bcebos.com/KL-SDK/klsdk-dev_paddle")
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

if (NOT XPU_XDNN_URL)
  set (XPU_XDNN_URL "${XPU_SDK_URL}/xdnn-${XPU_SDK_ENV}.tar.gz")
endif ()
message (STATUS "XPU_XDNN_URL: ${XPU_XDNN_URL}")
if (NOT XPU_XRE_URL)
  set (XPU_XRE_URL "${XPU_SDK_URL}/xre-${XPU_SDK_ENV}.tar.gz")
endif ()
message (STATUS "XPU_XRE_URL: ${XPU_XRE_URL}")

macro (prepare_xpu_sdk sdk sdk_url)
  ExternalProject_Add (
    extern_xpu_${sdk}
    ${EXTERNAL_PROJECT_LOG_ARGS}
    DOWNLOAD_DIR            ${XPU_DOWNLOAD_DIR}
    DOWNLOAD_COMMAND        wget --no-check-certificate -c -q ${sdk_url} -O ${sdk}.tar.gz && tar xf ${sdk}.tar.gz
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

if(XPU_WITH_XFT)
  if(XPU_XFT_ROOT)
    set (xpu_xft_root        "${XPU_XFT_ROOT}"  CACHE PATH "xpu xft include directory" FORCE)
    set (xpu_xft_include_dir "${XPU_XFT_ROOT}/include" CACHE PATH "xpu xft include directory" FORCE)

    include_directories (${xpu_xft_include_dir})
    add_library (xft SHARED IMPORTED GLOBAL)
    set_property (TARGET xft PROPERTY IMPORTED_LOCATION "${xpu_xft_root}/so/libxft.so")
    link_libraries (xft)
  else()
    if(NOT XPU_XFT_URL)
      set(XPU_XFT_URL "https://klx-sdk-release-public.su.bcebos.com/xft/dev/latest/")
    endif()
    if(NOT XPU_XFT_ENV)
      message(FATAL_ERROR "XPU_WITH_XFT=ON and XPU_XFT_ENV can not be null!")
    endif()

    ExternalProject_Add (
      extern_xpu_xft
      ${EXTERNAL_PROJECT_LOG_ARGS}
      DOWNLOAD_DIR            ${XPU_DOWNLOAD_DIR}
      DOWNLOAD_COMMAND        wget --no-check-certificate -c -q ${XPU_XFT_URL}/xft_${XPU_XFT_ENV}.tar.gz -O xft.tar.gz && tar xf xft.tar.gz
      CONFIGURE_COMMAND       ""
      BUILD_COMMAND           ""
      UPDATE_COMMAND          ""
      INSTALL_COMMAND         ${CMAKE_COMMAND} -E copy_directory ${XPU_DOWNLOAD_DIR}/xft_${XPU_XFT_ENV} ${XPU_INSTALL_DIR}/xpu/xft
    )

    set (xpu_xft_root        "${XPU_INSTALL_DIR}/xpu/xft"  CACHE PATH "xpu xft include directory" FORCE)
    set (xpu_xft_include_dir "${xpu_xft_root}/include" CACHE PATH "xpu xft include directory" FORCE)

    include_directories (${xpu_xft_include_dir})
    add_library (xft SHARED IMPORTED GLOBAL)
    set_property (TARGET xft PROPERTY IMPORTED_LOCATION "${xpu_xft_root}/so/libxft.so")
    add_dependencies (xft extern_xpu_xft)
    link_libraries (xft)
  endif(XPU_XFT_ROOT)
endif(XPU_WITH_XFT)

if (NOT XPU_SDK_ROOT)
  prepare_xpu_sdk (xdnn ${XPU_XDNN_URL} xpuapi)
  prepare_xpu_sdk (xre ${XPU_XRE_URL} xpurt)
  set (XPUAPI_LIB "${XPU_INSTALL_DIR}/xpu/xdnn/so/libxpuapi.so" CACHE FILEPATH "libxpuapi.so" FORCE)
  set (XPURT_LIB  "${XPU_INSTALL_DIR}/xpu/xre/so/libxpurt.so" CACHE FILEPATH "libxpurt.so" FORCE)
  set (xpu_builder_libs xpuapi CACHE INTERNAL "xpu builder libs")
  set (xpu_runtime_libs xpurt CACHE INTERNAL "xpu runtime libs")
  return ()
endif ()

# **DEPRECATED**, use XPU_SDK_URL/XPU_SDK_ENV in the future
message (STATUS "XPU_SDK_ROOT: ${XPU_SDK_ROOT}")

if (MSVC)
  set (XPU_INCLUDE_DIR "${XPU_SDK_ROOT}/XTCL/include" CACHE PATH "xpu include directory" FORCE)
  set (XPUAPI_LIB      "${XPU_SDK_ROOT}/XTCL/lib/libxpuapi.lib" CACHE FILEPATH "libxpuapi.lib" FORCE)
  set (XPURT_LIB       "${XPU_SDK_ROOT}/XTCL/runtime/lib/libxpurt.lib" CACHE FILEPATH "libxpurt.lib" FORCE)
else ()
  set (XPU_INCLUDE_DIR "${XPU_SDK_ROOT}/XTCL/include" CACHE PATH "xpu include directory" FORCE)
  set (XPUAPI_LIB      "${XPU_SDK_ROOT}/XTCL/shlib/libxpuapi.so" CACHE FILEPATH "libxpuapi.so" FORCE)
  set (XPURT_LIB       "${XPU_SDK_ROOT}/XTCL/runtime/shlib/libxpurt.so" CACHE FILEPATH "libxpurt.so" FORCE)
endif ()
include_directories (${XPU_INCLUDE_DIR})


if (MSVC)
  add_library (xpuapi STATIC IMPORTED GLOBAL)
else ()
  add_library (xpuapi SHARED IMPORTED GLOBAL)
endif ()
set_property (TARGET xpuapi PROPERTY IMPORTED_LOCATION ${XPUAPI_LIB})

if (MSVC)
  add_library (xpurt STATIC IMPORTED GLOBAL)
else ()
  add_library (xpurt SHARED IMPORTED GLOBAL)
endif ()
set_property (TARGET xpurt PROPERTY IMPORTED_LOCATION ${XPURT_LIB})

set (xpu_runtime_libs xpuapi xpurt CACHE INTERNAL "xpu runtime libs")
set (xpu_builder_libs xpuapi xpurt CACHE INTERNAL "xpu builder libs")
