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

if(NOT XPU_SDK_ROOT)
    INCLUDE(ExternalProject)

    if(LITE_WITH_X86)
        set(XPU_URL "https://paddlelite-demo.bj.bcebos.com/devices/baidu/xpu_toolchain-centos6.3-x86_64-gcc8.2.0-latest.tar.gz")
    elseif(LITE_WITH_ARM)
        set(XPU_URL "https://paddlelite-demo.bj.bcebos.com/devices/baidu/xpu_toolchain-ubuntu18.04.4-cross_compiling-aarch64-gcc5.4-latest.tar.gz")
    else()
        message(FATAL_ERROR "xpu doesn't supported the host device")
    endif()

    set(XPU_SOURCE_DIR "${THIRD_PARTY_PATH}/xpu")

    ExternalProject_Add(
        extern_xpu_sdk
        ${EXTERNAL_PROJECT_LOG_ARGS}
        DOWNLOAD_DIR          ${XPU_SOURCE_DIR}
        DOWNLOAD_COMMAND      wget --no-check-certificate -c -q -O xpu_toolchain.tar.gz ${XPU_URL} && tar xf xpu_toolchain.tar.gz
        CONFIGURE_COMMAND     ""
        BUILD_COMMAND         ""
        UPDATE_COMMAND        ""
        INSTALL_COMMAND       ""
    )

    set(XPU_SDK_ROOT ${XPU_SOURCE_DIR}/xpu_toolchain)
endif()

message(STATUS "XPU_SDK_ROOT: ${XPU_SDK_ROOT}")

set(XPU_XTDK_INCLUDE_DIR    "${XPU_SDK_ROOT}/XTDK/include" CACHE PATH "xpu xtdk include directory" FORCE)
set(XPUAPI_LIB              "${XPU_SDK_ROOT}/XTDK/shlib/libxpuapi.so" CACHE FILEPATH "libxpuapi.so" FORCE)
set(XPURT_LIB               "${XPU_SDK_ROOT}/XTDK/runtime/shlib/libxpurt.so" CACHE FILEPATH "libxpurt.so" FORCE)

INCLUDE_DIRECTORIES(${XPU_XTDK_INCLUDE_DIR})

ADD_LIBRARY(xpuapi SHARED IMPORTED GLOBAL)
SET_PROPERTY(TARGET xpuapi PROPERTY IMPORTED_LOCATION ${XPUAPI_LIB})
ADD_DEPENDENCIES(xpuapi extern_xpu_sdk)

ADD_LIBRARY(xpurt SHARED IMPORTED GLOBAL)
SET_PROPERTY(TARGET xpurt PROPERTY IMPORTED_LOCATION ${XPURT_LIB})
ADD_DEPENDENCIES(xpurt extern_xpu_sdk)

set(xpu_runtime_libs xpuapi xpurt CACHE INTERNAL "xpu runtime libs")
set(xpu_builder_libs xpuapi xpurt CACHE INTERNAL "xpu builder libs")

if(LITE_WITH_XTCL)
    set(XPU_XTCL_INCLUDE_DIR  "${XPU_SDK_ROOT}/XTCL/include" CACHE PATH "xpu xtcl include directory" FORCE)
    set(XTCL_LIB              "${XPU_SDK_ROOT}/XTCL/lib/libxtcl.a" CACHE FILEPATH "libxtcl.a" FORCE)
    set(TVM_LIB               "${XPU_SDK_ROOT}/XTCL/shlib/libtvm.so" CACHE FILEPATH "libtvm.so" FORCE)
    set(LLVM_10_LIB            "${XPU_SDK_ROOT}/XTCL/shlib/libLLVM-10.so" CACHE FILEPATH "libLLVM-8.so" FORCE)
    set(XPUJITC_LIB           "${XPU_SDK_ROOT}/XTCL/shlib/libxpujitc.so" CACHE FILEPATH "libxpujitc.so" FORCE)

    INCLUDE_DIRECTORIES(${XPU_XTCL_INCLUDE_DIR})

    ADD_LIBRARY(xtcl SHARED IMPORTED GLOBAL)
    SET_PROPERTY(TARGET xtcl PROPERTY IMPORTED_LOCATION ${XTCL_LIB})
    ADD_DEPENDENCIES(xtcl extern_xpu_sdk)

    ADD_LIBRARY(tvm SHARED IMPORTED GLOBAL)
    SET_PROPERTY(TARGET tvm PROPERTY IMPORTED_LOCATION ${TVM_LIB})
    ADD_DEPENDENCIES(tvm extern_xpu_sdk)

    ADD_LIBRARY(llvm_10 SHARED IMPORTED GLOBAL)
    SET_PROPERTY(TARGET llvm_10 PROPERTY IMPORTED_LOCATION ${LLVM_10_LIB})
    ADD_DEPENDENCIES(llvm_10 extern_xpu_sdk)

    ADD_LIBRARY(xpujitc SHARED IMPORTED GLOBAL)
    SET_PROPERTY(TARGET xpujitc PROPERTY IMPORTED_LOCATION ${XPUJITC_LIB})
    ADD_DEPENDENCIES(xpujitc extern_xpu_sdk)

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DDMLC_USE_GLOG=1")

    set(xpu_runtime_libs xtcl tvm xpuapi xpurt llvm_10 xpujitc CACHE INTERNAL "xpu runtime libs")
    set(xpu_builder_libs xtcl tvm xpuapi xpurt llvm_10 xpujitc CACHE INTERNAL "xpu builder libs")
endif()
