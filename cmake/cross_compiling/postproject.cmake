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

if(NOT LITE_WITH_LIGHT_WEIGHT_FRAMEWORK)
    return()
endif()

include(CheckCXXCompilerFlag)

if(ANDROID)
    include(cross_compiling/findar)
    
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -llog -fPIC")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -llog -fPIC")
endif()

if(ARMLINUX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
    if(ARMLINUX_ARCH_ABI STREQUAL "armv8")
        set(CMAKE_CXX_FLAGS "-march=armv8-a ${CMAKE_CXX_FLAGS}")
        set(CMAKE_C_FLAGS "-march=armv8-a ${CMAKE_C_FLAGS}")
        message(STATUS "NEON is enabled on arm64-v8a")
    endif()

    if(ARMLINUX_ARCH_ABI STREQUAL "armv7")
        set(CMAKE_CXX_FLAGS "-march=armv7-a -mfloat-abi=softfp -mfpu=neon-vfpv4 ${CMAKE_CXX_FLAGS}")
        set(CMAKE_C_FLAGS "-march=armv7-a -mfloat-abi=softfp -mfpu=neon-vfpv4 ${CMAKE_C_FLAGS}")
        message(STATUS "NEON is enabled on arm-v7a with softfp")
    endif()

    if(ARMLINUX_ARCH_ABI STREQUAL "armv7hf")
        set(CMAKE_CXX_FLAGS "-march=armv7-a -mfloat-abi=hard -mfpu=neon-vfpv4 ${CMAKE_CXX_FLAGS}")
        set(CMAKE_C_FLAGS "-march=armv7-a -mfloat-abi=hard -mfpu=neon-vfpv4 ${CMAKE_C_FLAGS}" )
        message(STATUS "NEON is enabled on arm-v7a with hard float")
    endif()
endif()

function(check_linker_flag)
    foreach(flag ${ARGN})
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${flag}")
        check_cxx_compiler_flag("" out_var)
        if(${out_var})
            set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${flag}")
        endif()
    endforeach()
    set(CMAKE_SHARED_LINKER_FLAGS ${CMAKE_SHARED_LINKER_FLAGS} PARENT_SCOPE)
endfunction()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
if (LITE_ON_TINY_PUBLISH)
    if((NOT LITE_WITH_PYTHON))
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-exceptions")
    endif()
    if(LITE_WITH_OPENCL AND (ARM_TARGET_LANG STREQUAL "clang"))
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fexceptions")
    endif()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffast-math -Ofast -Os -fomit-frame-pointer -fno-asynchronous-unwind-tables -fno-unwind-tables")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden -fvisibility-inlines-hidden -ffunction-sections")
    check_linker_flag(-Wl,--gc-sections)
endif()

if(LITE_WITH_OPENMP)
    find_package(OpenMP REQUIRED)
    if(OPENMP_FOUND OR OpenMP_CXX_FOUND)
        add_definitions(-DARM_WITH_OMP)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        message(STATUS "Found OpenMP ${OpenMP_VERSION} ${OpenMP_CXX_VERSION}")
        message(STATUS "OpenMP C flags:  ${OpenMP_C_FLAGS}")
        message(STATUS "OpenMP CXX flags:  ${OpenMP_CXX_FLAGS}")
        message(STATUS "OpenMP OpenMP_CXX_LIB_NAMES:  ${OpenMP_CXX_LIB_NAMES}")
        message(STATUS "OpenMP OpenMP_CXX_LIBRARIES:  ${OpenMP_CXX_LIBRARIES}")
    else()
        message(FATAL_ERROR "Could not found OpenMP!")
    endif()
endif()

# third party cmake args
set(CROSS_COMPILE_CMAKE_ARGS
    "-DCMAKE_SYSTEM_NAME=${CMAKE_SYSTEM_NAME}"
    "-DCMAKE_SYSTEM_VERSION=${CMAKE_SYSTEM_VERSION}")

if(ANDROID)
    set(CROSS_COMPILE_CMAKE_ARGS ${CROSS_COMPILE_CMAKE_ARGS}
        "-DCMAKE_ANDROID_ARCH_ABI=${CMAKE_ANDROID_ARCH_ABI}"
        "-DCMAKE_ANDROID_NDK=${CMAKE_ANDROID_NDK}"
        "-DCMAKE_ANDROID_STL_TYPE=${CMAKE_ANDROID_STL_TYPE}"
        "-DCMAKE_ANDROID_NDK_TOOLCHAIN_VERSION=${CMAKE_ANDROID_NDK_TOOLCHAIN_VERSION}")
endif()
  
if(IOS)
    set(CROSS_COMPILE_CMAKE_ARGS ${CROSS_COMPILE_CMAKE_ARGS}
        "-DCMAKE_OSX_ARCHITECTURES=${CMAKE_OSX_ARCHITECTURES}"
        "-DCMAKE_SYSTEM_PROCESSOR=${CMAKE_SYSTEM_PROCESSOR}"
        "-DCMAKE_OSX_SYSROOT=${CMAKE_OSX_SYSROOT}")
endif()
