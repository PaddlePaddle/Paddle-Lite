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

if(NOT LITE_WITH_XPU)
    return()
endif()

set(XPU_SOURCE_DIR ${THIRD_PARTY_PATH}/xpu)
set(XPU_DOWNLOAD_DIR ${XPU_SOURCE_DIR}/download)
set(XPU_INSTALL_DIR ${THIRD_PARTY_PATH}/install)

if(NOT XPU_SDK_ROOT)
    INCLUDE(ExternalProject)

    if(NOT XPU_SDK_URL)
        set(XPU_SDK_URL "https://baidu-kunlun-product.cdn.bcebos.com/KL-SDK/klsdk-dev_paddle")
    endif()

    if(NOT XPU_SDK_ENV)
        if(LITE_WITH_X86)
            set(XPU_SDK_ENV "bdcentos_x86_64")
        elseif(LITE_WITH_ARM)
            set(XPU_SDK_ENV "kylin_aarch64")
        else()
            message(FATAL_ERROR "xpu doesn't supported the host env")
        endif()
    endif()

    # get xre from XPU_XRE_URL
    set(XPU_XRE_URL "${XPU_SDK_URL}/xre-${XPU_SDK_ENV}.tar.gz")
    message(STATUS "XPU_XRE_URL: ${XPU_XRE_URL}")

    ExternalProject_Add(
        extern_xpu_xre
        ${EXTERNAL_PROJECT_LOG_ARGS}
        DOWNLOAD_DIR            ${XPU_DOWNLOAD_DIR}
        DOWNLOAD_COMMAND        wget --no-check-certificate -c -q ${XPU_XRE_URL} && tar xf xre-${XPU_SDK_ENV}.tar.gz
        CONFIGURE_COMMAND       ""
        BUILD_COMMAND           ""
        UPDATE_COMMAND          ""
        INSTALL_COMMAND         ${CMAKE_COMMAND} -E copy_directory ${XPU_DOWNLOAD_DIR}/xre-${XPU_SDK_ENV} ${XPU_INSTALL_DIR}/xpu/xre
    )

    set(XPU_XRE_ROOT            "${XPU_INSTALL_DIR}/xpu/xre"  CACHE PATH "xpu xre include directory" FORCE)
    set(XPU_XRE_INCLUDE_DIR     "${XPU_XRE_ROOT}/include" CACHE PATH "xpu xre include directory" FORCE)
    set(XPURT_LIB               "${XPU_XRE_ROOT}/so/libxpurt.so" CACHE FILEPATH "libxpurt.so" FORCE)

    INCLUDE_DIRECTORIES(${XPU_XRE_INCLUDE_DIR})

    ADD_LIBRARY(xpurt SHARED IMPORTED GLOBAL)
    SET_PROPERTY(TARGET xpurt PROPERTY IMPORTED_LOCATION ${XPURT_LIB})
    ADD_DEPENDENCIES(xpurt extern_xpu_xre)

    set(xpu_runtime_libs xpurt CACHE INTERNAL "xpu runtime libs")

    # get xdnn from XPU_XDNN_URL
    set(XPU_XDNN_URL "${XPU_SDK_URL}/xdnn-${XPU_SDK_ENV}.tar.gz")
    message(STATUS "XPU_XDNN_URL: ${XPU_XDNN_URL}")

    ExternalProject_Add(
        extern_xpu_xdnn
        ${EXTERNAL_PROJECT_LOG_ARGS}
        DOWNLOAD_DIR            ${XPU_DOWNLOAD_DIR}
        DOWNLOAD_COMMAND        wget --no-check-certificate -c -q ${XPU_XDNN_URL} && tar xf xdnn-${XPU_SDK_ENV}.tar.gz
        CONFIGURE_COMMAND       ""
        BUILD_COMMAND           ""
        UPDATE_COMMAND          ""
        INSTALL_COMMAND         ${CMAKE_COMMAND} -E copy_directory ${XPU_DOWNLOAD_DIR}/xdnn-${XPU_SDK_ENV} ${XPU_INSTALL_DIR}/xpu/xdnn
    )

    set(XPU_XDNN_ROOT           "${XPU_INSTALL_DIR}/xpu/xdnn" CACHE PATH "xpu xdnn root" FORCE)
    set(XPU_XDNN_INCLUDE_DIR    "${XPU_XDNN_ROOT}/include" CACHE PATH "xpu xdnn include directory" FORCE)
    set(XPUAPI_LIB              "${XPU_XDNN_ROOT}/so/libxpuapi.so" CACHE FILEPATH "libxpuapi.so" FORCE)

    INCLUDE_DIRECTORIES(${XPU_XDNN_INCLUDE_DIR})

    ADD_LIBRARY(xpuapi SHARED IMPORTED GLOBAL)
    SET_PROPERTY(TARGET xpuapi PROPERTY IMPORTED_LOCATION ${XPUAPI_LIB})
    ADD_DEPENDENCIES(xpuapi extern_xpu_xdnn extern_xpu_xre)

    set(xpu_builder_libs xpuapi CACHE INTERNAL "xpu builder libs")

    return()
endif()

message(STATUS "XPU_SDK_ROOT: ${XPU_SDK_ROOT}")

set(XPU_XTDK_INCLUDE_DIR    "${XPU_SDK_ROOT}/XTDK/include" CACHE PATH "xpu xtdk include directory" FORCE)
set(XPUAPI_LIB              "${XPU_SDK_ROOT}/XTDK/shlib/libxpuapi.so" CACHE FILEPATH "libxpuapi.so" FORCE)
set(XPURT_LIB               "${XPU_SDK_ROOT}/XTDK/runtime/shlib/libxpurt.so" CACHE FILEPATH "libxpurt.so" FORCE)

INCLUDE_DIRECTORIES(${XPU_XTDK_INCLUDE_DIR})

ADD_LIBRARY(xpuapi SHARED IMPORTED GLOBAL)
SET_PROPERTY(TARGET xpuapi PROPERTY IMPORTED_LOCATION ${XPUAPI_LIB})

ADD_LIBRARY(xpurt SHARED IMPORTED GLOBAL)
SET_PROPERTY(TARGET xpurt PROPERTY IMPORTED_LOCATION ${XPURT_LIB})

set(xpu_runtime_libs xpuapi xpurt CACHE INTERNAL "xpu runtime libs")
set(xpu_builder_libs xpuapi xpurt CACHE INTERNAL "xpu builder libs")

if(LITE_WITH_XTCL)
    set(XPU_XTCL_INCLUDE_DIR  "${XPU_SDK_ROOT}/XTCL/include" CACHE PATH "xpu xtcl include directory" FORCE)
    set(XTCL_LIB              "${XPU_SDK_ROOT}/XTCL/lib/libxtcl.a" CACHE FILEPATH "libxtcl.a" FORCE)
    set(TVM_LIB               "${XPU_SDK_ROOT}/XTCL/shlib/libtvm.so" CACHE FILEPATH "libtvm.so" FORCE)
    set(LLVM_8_LIB            "${XPU_SDK_ROOT}/XTCL/shlib/libLLVM-8.so" CACHE FILEPATH "libLLVM-8.so" FORCE)
    set(XPUJITC_LIB           "${XPU_SDK_ROOT}/XTCL/shlib/libxpujitc.so" CACHE FILEPATH "libxpujitc.so" FORCE)

    INCLUDE_DIRECTORIES(${XPU_XTCL_INCLUDE_DIR})

    ADD_LIBRARY(xtcl SHARED IMPORTED GLOBAL)
    SET_PROPERTY(TARGET xtcl PROPERTY IMPORTED_LOCATION ${XTCL_LIB})

    ADD_LIBRARY(tvm SHARED IMPORTED GLOBAL)
    SET_PROPERTY(TARGET tvm PROPERTY IMPORTED_LOCATION ${TVM_LIB})

    ADD_LIBRARY(llvm_8 SHARED IMPORTED GLOBAL)
    SET_PROPERTY(TARGET llvm_8 PROPERTY IMPORTED_LOCATION ${LLVM_8_LIB})

    ADD_LIBRARY(xpujitc SHARED IMPORTED GLOBAL)
    SET_PROPERTY(TARGET xpujitc PROPERTY IMPORTED_LOCATION ${XPUJITC_LIB})

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DDMLC_USE_GLOG=1")

    set(xpu_runtime_libs xtcl tvm xpuapi xpurt llvm_8 xpujitc CACHE INTERNAL "xpu runtime libs")
    set(xpu_builder_libs xtcl tvm xpuapi xpurt llvm_8 xpujitc CACHE INTERNAL "xpu builder libs")
endif()
