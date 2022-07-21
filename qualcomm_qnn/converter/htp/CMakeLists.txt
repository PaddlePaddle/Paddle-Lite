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
cmake_minimum_required(VERSION 3.10)

if(NOT DEFINED NNADAPTER_QUALCOMM_HEXAGON_TOOLS_ROOT)
  set(NNADAPTER_QUALCOMM_HEXAGON_TOOLS_ROOT $ENV{NNADAPTER_QUALCOMM_HEXAGON_TOOLS_ROOT})
endif()
message(STATUS "NNADAPTER_QUALCOMM_HEXAGON_TOOLS_ROOT: ${NNADAPTER_QUALCOMM_HEXAGON_TOOLS_ROOT}")
if(NOT NNADAPTER_QUALCOMM_HEXAGON_TOOLS_ROOT)
  message(FATAL_ERROR "Must set NNADAPTER_QUALCOMM_HEXAGON_TOOLS_ROOT or env NNADAPTER_QUALCOMM_HEXAGON_TOOLS_ROOT when NNADAPTER_WITH_QUALCOMM_QNN=ON")
endif()

set(HTP_OP_PACKAGE_NAME qualcomm_qnn_htp_custom_op_package)
set(PKG_NAME "Custom.Htp.OpPackage")

# Resolve the compilation error caused by the introduction of HTP header files
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_FLAGS "")
set(CMAKE_CXX_FLAGS "-std=c++17 -fPIC -Os -Wl,--gc-sections -Wl,--allow-shlib-undefined -Wl,--strip-all -fvisibility=hidden -fvisibility-inlines-hidden -fexceptions -fasynchronous-unwind-tables -funwind-tables -Wreorder -Wno-missing-braces -Wno-format")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I${NNADAPTER_QUALCOMM_QNN_SDK_ROOT}/include -DTHIS_PKG_NAME=${PKG_NAME} -MMD")

if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wl,-exclude-libs,ALL -D__HVXDBL__ -DUSE_OS_LINUX -fomit-frame-pointer -Wno-invalid-offsetof -stdlib=libc++")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I${NNADAPTER_QUALCOMM_HEXAGON_TOOLS_ROOT}/libnative/include")
  set(LD_FLAGS "-Wl,--whole-archive -L${NNADAPTER_QUALCOMM_HEXAGON_TOOLS_ROOT}/libnative/lib -lnative")
  set(LD_FLAGS "${LD_FLAGS} -Wl,--whole-archive -L${NNADAPTER_QUALCOMM_QNN_SDK_ROOT}/target/x86_64-linux-clang/lib -lQnnHtp")
  set(LD_FLAGS "${LD_FLAGS} -Wl,--no-whole-archive -lpthread")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Android")
  if(NOT DEFINED NNADAPTER_ANDROID_NDK_ROOT)
    set(NNADAPTER_ANDROID_NDK_ROOT $ENV{NNADAPTER_ANDROID_NDK_ROOT})
  endif()
  message(STATUS "NNADAPTER_ANDROID_NDK_ROOT: ${NNADAPTER_ANDROID_NDK_ROOT}")
  if(NOT NNADAPTER_ANDROID_NDK_ROOT)
    message(FATAL_ERROR "Must set NNADAPTER_ANDROID_NDK_ROOT or env NNADAPTER_ANDROID_NDK_ROOT when compile in android os")
  endif()
  set(ANDROID_ARM_CLANG_OPT "ARM_CLANG_OPTS =--target=aarch64-none-linux-android21 --sysroot=${NNADAPTER_ANDROID_NDK_ROOT}/toolchains/llvm/prebuilt/linux-x86_64/sysroot")
  set(CMAKE_C_COMPILER ${ANDROID_NDK_ROOT}/toolchains/llvm/prebuilt/linux-x86_64/bin/clang ${ANDROID_ARM_CLANG_OPT})
  set(CMAKE_CXX_COMPILER ${ANDROID_NDK_ROOT}/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++ ${ANDROID_ARM_CLANG_OPT})
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__HVXDBL__ -DUSE_OS_LINUX -DANDROID -fomit-frame-pointer -Wno-invalid-offsetof -stdlib=libc++")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I${NNADAPTER_QUALCOMM_HEXAGON_TOOLS_ROOT}/libnative/include")
  set(LD_FLAGS "-L${NNADAPTER_QUALCOMM_QNN_SDK_ROOT}/target/aarch64-android/lib -lQnnHtp -lQnnHtpPrepare")
elseif(CMAKE_SYSTEM_NAME STREQUAL "QNX")
  string(REGEX REPLACE "(.*)/tools/HEXAGON_Tools/(.*)" "\\1" HEXAGON_SDK_ROOT ${NNADAPTER_QUALCOMM_HEXAGON_TOOLS_ROOT})
  set(CMAKE_C_COMPILER ${NNADAPTER_QUALCOMM_HEXAGON_TOOLS_ROOT}/bin/hexagon-clang)
  set(CMAKE_CXX_COMPILER ${NNADAPTER_QUALCOMM_HEXAGON_TOOLS_ROOT}/bin/hexagon-clang++)
  set(CMAKE_C_COMPILER_TARGET "")
  set(CMAKE_CXX_COMPILER_TARGET "")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mv68 -mhvx -mhvx-length=128B -mhmx -DUSE_OS_QURT -O2 -Wno-reorder")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I${HEXAGON_SDK_ROOT}/rtos/qurt/computev68/include/qurt")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I${HEXAGON_SDK_ROOT}/incs")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I${HEXAGON_SDK_ROOT}/incs/stddef")
  set(LD_FLAGS "")
else()
  message(FATAL_ERROR "The os arch is not supported")
endif()

include_directories(.)
include_directories(${NNADAPTER_QUALCOMM_QNN_SDK_ROOT}/include/HTP/)
include_directories(${NNADAPTER_QUALCOMM_QNN_SDK_ROOT}/include/HTP/core/)

aux_source_directory(. SRCS)
add_library(${HTP_OP_PACKAGE_NAME} SHARED ${SRCS})
target_link_libraries(${HTP_OP_PACKAGE_NAME} "${LD_FLAGS}")

# Copy libs
string(REGEX REPLACE "(.*)/converter/htp" "\\1" QUALCOMM_QNN_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
add_custom_command(TARGET ${HTP_OP_PACKAGE_NAME}
  POST_BUILD
  COMMAND cp -r "${CMAKE_CURRENT_BINARY_DIR}/*.so" "${QUALCOMM_QNN_BINARY_DIR}"
  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/lib${HTP_OP_PACKAGE_NAME}.so
)
