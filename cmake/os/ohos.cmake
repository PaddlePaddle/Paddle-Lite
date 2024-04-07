# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

if (NOT DEFINED OHOS_SDK)
    set(OHOS_SDK $ENV{OHOS_SDK})
    if (NOT OHOS_SDK)
        message(FATAL_ERROR "Must set OHOS_SDK")
    endif ()
endif ()
message(STATUS "Build with OHOS")

# Definitions
add_definitions(-DLITE_WITH_LINUX)
add_definitions(-DLITE_WITH_OHOS)
set(OHOS_PLATFORM "OHOS")
set(ARM_TARGET_OS "ohos")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}-Wno-error=register -Wno-unused-but-set-variable")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
set(CMAKE_CXX_STANDARD 14)

if (ARM_TARGET_ARCH_ABI STREQUAL "armv7")
    set(OHOS_ARCH "armeabi-v7a")
endif ()

if (ARM_TARGET_ARCH_ABI STREQUAL "armv8")
    set(OHOS_ARCH "arm64-v8a")
endif ()

# if not set ARM_TARGET_ARCH_ABI default to armv8
if (NOT DEFINED ARM_TARGET_ARCH_ABI)
    message(STATUS "ARM_TARGET_ARCH_ABI not defined, use armv8")
    set(ARM_TARGET_ARCH_ABI "armv8")
endif ()

if (NOT DEFINED OHOS_ARCH)
    message(STATUS "OHOS_ARCH not defined, use arm64-v8a")
    set(OHOS_ARCH "arm64-v8a")
endif ()

if (NOT DEFINED OHOS_STL)
    message(STATUS "OHOS_STL not defined, use c++_shared")
    set(OHOS_STL "c++_shared")
endif ()


# print all config
message(STATUS "OHOS_SDK: ${OHOS_SDK}")
message(STATUS "OHOS_ARCH: ${OHOS_ARCH}")
message(STATUS "OHOS_STL: ${OHOS_STL}")
message(STATUS "OHOS_PLATFORM: ${OHOS_PLATFORM}")
message(STATUS "OHOS_TOOLCHAIN: ${OHOS_TOOLCHAIN}")


if (LITE_WITH_LOG)
    set(OHOS_LINKED_LIBS ${OHOS_LINKED_LIBS} libhilog_ndk.z.so)
    message(STATUS "OHOS_LINKED_LIBS: ${OHOS_LINKED_LIBS}")
endif ()
