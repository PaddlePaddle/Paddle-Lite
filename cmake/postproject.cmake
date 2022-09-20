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

if(NOT LITE_WITH_ARM)
    return()
endif()
include(CheckCXXCompilerFlag)
if(ANDROID)
    include(findar)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -llog -fPIC")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -llog -fPIC")
    if(LITE_WITH_ARM82_FP16)
        if(${ANDROID_NDK_MAJOR})
            if(${ANDROID_NDK_MAJOR} GREATER "17")
                if (${ARM_TARGET_ARCH_ABI} STREQUAL "armv8")
                  set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   -march=armv8.2-a+fp16+nolse")
                  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8.2-a+fp16+nolse")
                elseif(${ARM_TARGET_ARCH_ABI} STREQUAL "armv7")
                  if(${ANDROID_NDK_MAJOR} GREATER "21")
                    set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   -march=armv8.2-a+fp16 -mfpu=neon-fp-armv8 -mfloat-abi=softfp")
                    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8.2-a+fp16 -mfpu=neon-fp-armv8 -mfloat-abi=softfp")
                  else()
                    # suggested to use ndk r22 or newer version to be compatible with armv7 fp16 intrinsic func compilation
                    message(FATAL_ERROR "NDK VERSION: ${ANDROID_NDK_MAJOR}, however it must be greater than 21 when arm v7 fp16 is ON")
                  endif()
                endif()
            else()
                message(FATAL_ERROR "NDK VERSION: ${ANDROID_NDK_MAJOR}, however it must be greater than 17 when arm fp16 is ON")
            endif()
        endif()
    endif()

    if(LITE_WITH_ARM8_SVE2)
        if ((ARM_TARGET_ARCH_ABI STREQUAL "armv8"))
          if (${ANDROID_NDK_MAJOR})
            if(${ANDROID_NDK_MAJOR} GREATER_EQUAL "23")
                set(CMAKE_C_FLAGS    "${CMAKE_C_FLAGS} -march=armv8.2-a+sve2+fp16+dotprod+f32mm+i8mm+nolse")
                set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -march=armv8.2-a+sve2+fp16+dotprod+f32mm+i8mm+nolse")
            else()
                message(FATAL_ERROR "NDK VERSION: ${ANDROID_NDK_MAJOR}, however it must be greater equal 23 when sve2 is ON")
            endif()
          endif()
        else()
        message(FATAL_ERROR "The arm_abi is ${ARM_TARGET_ARCH_ABI}, the arm_abi must be armv8 when sve2 is ON")
        endif()
    endif()

    if(LITE_WITH_ARM82_INT8_SDOT)
        if(${ANDROID_NDK_MAJOR})
            if(${ANDROID_NDK_MAJOR} GREATER "17")
                set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   -march=armv8.2-a+dotprod+nolse")
                set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8.2-a+dotprod+nolse")
            endif()
        endif()
    endif()

    # Don't re-export libgcc symbols
    set(REMOVE_ATOMIC_GCC_SYMBOLS "-Wl,--exclude-libs,libatomic.a -Wl,--exclude-libs,libgcc.a")
    set(CMAKE_SHARED_LINKER_FLAGS "${REMOVE_ATOMIC_GCC_SYMBOLS} ${CMAKE_SHARED_LINKER_FLAGS}")
    set(CMAKE_MODULE_LINKER_FLAGS "${REMOVE_ATOMIC_GCC_SYMBOLS} ${CMAKE_MODULE_LINKER_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${REMOVE_ATOMIC_GCC_SYMBOLS} ${CMAKE_EXE_LINKER_FLAGS}")

    # Only the libunwind.a from clang(with libc++) provide C++ exception handling support for 32-bit ARM
    # Refer to https://android.googlesource.com/platform/ndk/+/master/docs/BuildSystemMaintainers.md#Unwinding
    if (ARM_TARGET_LANG STREQUAL "clang" AND ARM_TARGET_ARCH_ABI STREQUAL "armv7" AND ANDROID_STL_TYPE MATCHES "^c\\+\\+_")
        set(REMOVE_UNWIND_SYMBOLS "-Wl,--exclude-libs,libunwind.a")
        set(CMAKE_SHARED_LINKER_FLAGS "${REMOVE_UNWIND_SYMBOLS} ${CMAKE_SHARED_LINKER_FLAGS}")
        set(CMAKE_MODULE_LINKER_FLAGS "${REMOVE_UNWIND_SYMBOLS} ${CMAKE_MODULE_LINKER_FLAGS}")
        set(CMAKE_EXE_LINKER_FLAGS "${REMOVE_UNWIND_SYMBOLS} ${CMAKE_EXE_LINKER_FLAGS}")
    endif()
endif()
if(ARMLINUX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
    if(ARMLINUX_ARCH_ABI STREQUAL "armv8")
        set(CMAKE_CXX_FLAGS "-march=armv8-a ${CMAKE_CXX_FLAGS}")
        set(CMAKE_C_FLAGS "-march=armv8-a ${CMAKE_C_FLAGS}")
        message(STATUS "NEON is enabled on arm64-v8a")
        if (LITE_WITH_ARM82_FP16)
          set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8.2-a+fp16")
          set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   -march=armv8.2-a+fp16")
        endif()
    endif()

    if(ARMLINUX_ARCH_ABI STREQUAL "armv7")
        set(CMAKE_CXX_FLAGS "-march=armv7-a -mfloat-abi=softfp -mfpu=neon ${CMAKE_CXX_FLAGS}")
        set(CMAKE_C_FLAGS "-march=armv7-a -mfloat-abi=softfp -mfpu=neon ${CMAKE_C_FLAGS}")
        message(STATUS "NEON is enabled on arm-v7a with softfp")
        if (LITE_WITH_ARM82_FP16)
          set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8.2-a+fp16 -mfpu=neon-fp-armv8")
          set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   -march=armv8.2-a+fp16 -mfpu=neon-fp-armv8")
        endif()
    endif()

    if(ARMLINUX_ARCH_ABI STREQUAL "armv7hf")
        set(CMAKE_CXX_FLAGS "-march=armv7-a -mfloat-abi=hard -mfpu=neon ${CMAKE_CXX_FLAGS}")
        set(CMAKE_C_FLAGS "-march=armv7-a -mfloat-abi=hard -mfpu=neon ${CMAKE_C_FLAGS}" )
        message(STATUS "NEON is enabled on arm-v7a with hard float")
        if (LITE_WITH_ARM82_FP16)
          set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8.2-a+fp16 -mfpu=neon-fp-armv8")
          set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   -march=armv8.2-a+fp16 -mfpu=neon-fp-armv8")
        endif()
    endif()
endif()

if(QNX)
    add_definitions(-DLITE_WITH_QNX)
    add_definitions(-D_QNX_SOURCE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -V${QNX_COMPILER_TARGET} -fPIC -D_QNX_SOURCE=1")
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
if((LITE_WITH_OPENCL AND (ARM_TARGET_LANG STREQUAL "clang")) OR LITE_WITH_PYTHON OR LITE_WITH_EXCEPTION OR (NOT LITE_ON_TINY_PUBLISH))
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fexceptions -fasynchronous-unwind-tables -funwind-tables")
else ()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-exceptions -fno-asynchronous-unwind-tables -fno-unwind-tables")
endif()

if((LITE_WITH_METAL AND (ARM_TARGET_LANG STREQUAL "clang")) OR LITE_WITH_PYTHON OR LITE_WITH_EXCEPTION OR (NOT LITE_ON_TINY_PUBLISH))
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fexceptions -fasynchronous-unwind-tables -funwind-tables")
else ()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-exceptions -fno-asynchronous-unwind-tables -fno-unwind-tables")
endif()

if (LITE_ON_TINY_PUBLISH)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffast-math -Ofast -Os -fomit-frame-pointer")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden -fvisibility-inlines-hidden -ffunction-sections")
    # 1. strip useless symbols from third-party libs
    # exclude-libs is not supported on macOs system
    if(NOT ARMMACOS)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wl,--exclude-libs,ALL")
      check_linker_flag(-Wl,--gc-sections)
    endif()
    # 2. strip rtti lib to reduce lib size
    #     2.1 replace typeid by fastTypeId
    #     2.2 replace dynamic_cast by static_cast
    if ((NOT LITE_WITH_NNADAPTER) AND (NOT LITE_WITH_LOG))
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti")
    endif()
endif()

if(ARM_TARGET_LANG STREQUAL "clang")
    # note(ysh329): fix slow compilation for arm cpu, 
    #               and abnormal exit compilation for opencl due to lots of warning
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-inconsistent-missing-override -Wno-return-type")
endif()

message(STATUS "ANDROID_NDK_MAJOR: ${ANDROID_NDK_MAJOR}")

if(LITE_WITH_OPENMP)
    if (ARM_TARGET_LANG STREQUAL "gcc")
        set(OpenMP_C_FLAGS "-fopenmp")
        set(OpenMP_C_LIB_NAMES "omp")
        set(OpenMP_CXX_FLAGS "-fopenmp")
        set(OpenMP_CXX_LIB_NAMES "omp")
        set(OpenMP_omp_LIBRARY omp)
        set(OpenMP_C_FLAGS_WORK "-fopenmp")
        set(OpenMP_C_LIB_NAMES_WORK "omp")
        set(OpenMP_CXX_FLAGS_WORK "-fopenmp")
        set(OpenMP_CXX_LIB_NAMES_WORK "omp")
    endif()
    find_package(OpenMP REQUIRED)
    set(LIBOMP_ENABLE_SHARED OFF)
    if(OPENMP_FOUND OR OpenMP_CXX_FOUND)
        add_definitions(-DARM_WITH_OMP)
        if(${ANDROID_NDK_MAJOR})
            if(${ANDROID_NDK_MAJOR} GREATER 20)
                message(STATUS "ANDROID_NDK_MAJOR GREATER 20")
                set(OPENMP_LINK_FLAGS "-fopenmp -static-openmp")
            else()
                set(OPENMP_LINK_FLAGS "-fopenmp")
            endif()
        endif()
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OPENMP_LINK_FLAGS}")
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${OPENMP_LINK_FLAGS}")
        message(STATUS "Found OpenMP ${OpenMP_VERSION} ${OpenMP_CXX_VERSION}")
        message(STATUS "OpenMP C flags:  ${OpenMP_C_FLAGS}")
        message(STATUS "OpenMP CXX flags:  ${OpenMP_CXX_FLAGS}")
        message(STATUS "OpenMP EXE LINKER flags:  ${OPENMP_LINK_FLAGS}")
        message(STATUS "OpenMP OpenMP_CXX_LIB_NAMES:  ${OpenMP_CXX_LIB_NAMES}")
        message(STATUS "OpenMP OpenMP_CXX_LIBRARIES:  ${OpenMP_CXX_LIBRARIES}")
    else()
        message(FATAL_ERROR "Could not found OpenMP!")
    endif()
endif()

if (CMAKE_CXX_FLAGS)
string(REGEX REPLACE " \\-g " " " CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
endif()

if (CMAKE_C_FLAGS)
string(REGEX REPLACE " \\-g " " " CMAKE_C_FLAGS ${CMAKE_C_FLAGS})
endif ()


message(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
message(STATUS "CMAKE_C_FLAGS: ${CMAKE_C_FLAGS}")
# third party cmake args
set(CROSS_COMPILE_CMAKE_ARGS
    "-DCMAKE_SYSTEM_NAME=${CMAKE_SYSTEM_NAME}"
    "-DCMAKE_SYSTEM_VERSION=${CMAKE_SYSTEM_VERSION}")
if(ANDROID)
    set(CROSS_COMPILE_CMAKE_ARGS ${CROSS_COMPILE_CMAKE_ARGS}
        "-DCMAKE_ANDROID_ARCH_ABI=${CMAKE_ANDROID_ARCH_ABI}"
        "-DCMAKE_ANDROID_NDK=${CMAKE_ANDROID_NDK}"
        "-DCMAKE_ANDROID_STL_TYPE=${CMAKE_ANDROID_STL_TYPE}"
        "-DANDROID_ABI=${CMAKE_ANDROID_ARCH_ABI}"
        "-DANDROID_TOOLCHAIN=${ARM_TARGET_LANG}"
        "-DANDROID_STL=${CMAKE_ANDROID_STL_TYPE}"
        "-DCMAKE_SYSTEM_PROCESSOR=${CMAKE_SYSTEM_PROCESSOR}"
        "-DCMAKE_TOOLCHAIN_FILE=${CMAKE_ANDROID_NDK}/build/cmake/android.toolchain.cmake"
        "-DCMAKE_ANDROID_NDK_TOOLCHAIN_VERSION=${CMAKE_ANDROID_NDK_TOOLCHAIN_VERSION}"
        "-DANDROID_PLATFORM=android-${ANDROID_NATIVE_API_LEVEL}"
        "-D__ANDROID_API__=${ANDROID_NATIVE_API_LEVEL}"
        )
endif()
  
if(IOS)
    if(LITE_WITH_ARM82_FP16)
      set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   -march=armv8.2-a+fp16+nolse")
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8.2-a+fp16+nolse")
    endif()
    set(CROSS_COMPILE_CMAKE_ARGS ${CROSS_COMPILE_CMAKE_ARGS}
        "-DCMAKE_OSX_ARCHITECTURES=${CMAKE_OSX_ARCHITECTURES}"
        "-DCMAKE_SYSTEM_PROCESSOR=${CMAKE_SYSTEM_PROCESSOR}"
        "-DCMAKE_OSX_SYSROOT=${CMAKE_OSX_SYSROOT}")
endif()

if(ARMMACOS)
    set(CROSS_COMPILE_CMAKE_ARGS ${CROSS_COMPILE_CMAKE_ARGS}
        "-DCMAKE_OSX_ARCHITECTURES=${CMAKE_OSX_ARCHITECTURES}"
        "-DCMAKE_SYSTEM_PROCESSOR=${CMAKE_SYSTEM_PROCESSOR}"
        "-DCMAKE_OSX_SYSROOT=${CMAKE_OSX_SYSROOT}")
endif()
