# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set(ARMLINUX TRUE)
set(ARM_TARGET_ARCH_ABI_LIST "armv8" "armv7" "armv7hf")
set(CMAKE_SYSTEM_NAME Linux)

# ARMLINUX_ARCH_ABI
if(NOT DEFINED ARMLINUX_ARCH_ABI)
    set(ARMLINUX_ARCH_ABI ${ARM_TARGET_ARCH_ABI})
else()
    if(NOT ARM_TARGET_ARCH_ABI IN_LIST ARM_TARGET_ARCH_ABI_LIST)
        message(FATAL_ERROR "ARM_TARGET_ARCH_ABI should be one of ${ARM_TARGET_ARCH_ABI_LIST}")
    endif()
endif()

# Toolchain
if(ARMLINUX_ARCH_ABI STREQUAL "armv8")
    set(CMAKE_SYSTEM_PROCESSOR aarch64)
    set(CMAKE_C_COMPILER "aarch64-linux-gnu-gcc")
    set(CMAKE_CXX_COMPILER "aarch64-linux-gnu-g++")
endif()
if(ARMLINUX_ARCH_ABI STREQUAL "armv7")
    set(CMAKE_SYSTEM_PROCESSOR arm)
    set(CMAKE_C_COMPILER "arm-linux-gnueabi-gcc")
    set(CMAKE_CXX_COMPILER "arm-linux-gnueabi-g++")
endif()
if(ARMLINUX_ARCH_ABI STREQUAL "armv7hf")
    set(CMAKE_SYSTEM_PROCESSOR arm)
    set(CMAKE_C_COMPILER "arm-linux-gnueabihf-gcc")
    set(CMAKE_CXX_COMPILER "arm-linux-gnueabihf-g++")
endif()
set(HOST_C_COMPILER $ENV{CC})
set(HOST_CXX_COMPILER $ENV{CXX})
if(NOT ${HOST_C_COMPILER})
    set(CMAKE_C_COMPILER ${HOST_C_COMPILER})
endif()
if(NOT ${HOST_CXX_COMPILER})
    set(CMAKE_CXX_COMPILER ${HOST_CXX_COMPILER})
endif()
message(STATUS "armlinux CMAKE_C_COMPILER: ${CMAKE_C_COMPILER}")
message(STATUS "armlinux CMAKE_CXX_COMPILER: ${CMAKE_CXX_COMPILER}")

# Definitions
add_definitions(-DLITE_WITH_LINUX)
