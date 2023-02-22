# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

cmake_minimum_required(VERSION 3.0)

SET(AARCH32_SRCS)

SET(AARCH64_SRCS)

SET(NEON_FP16_SRCS)

add_subdirectory(relu)
add_subdirectory(concat)

SET_PROPERTY(SOURCE ${ARM_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " -marm ")
SET_PROPERTY(SOURCE ${ARMV6_VFP_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " -march=armv6 -mfpu=vfp -munaligned-access ")
SET_PROPERTY(SOURCE ${ARMV7A_NEON_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " -march=armv7-a -mfpu=neon ")
SET_PROPERTY(SOURCE ${ARMV7A_NEONFP16_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " -march=armv7-a -mfpu=neon-fp16 ")
IF(CMAKE_C_COMPILER_ID STREQUAL "GNU")
  SET_PROPERTY(SOURCE ${FP16_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " -mfp16-format=ieee ")
ENDIF()
SET_PROPERTY(SOURCE ${ARMV7A_NEONFMA_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " -march=armv7-a -mfpu=neon-vfpv4 ")
SET_PROPERTY(SOURCE ${ARMV8A_NEONV8_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " -march=armv8-a -mfpu=neon-fp-armv8 ")
SET_PROPERTY(SOURCE ${ARMV82A_FP16_NEONV8_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " -march=armv8.2-a+fp16 -mfpu=neon-fp-armv8 -fno-math-errno -ffinite-math-only ")
SET_PROPERTY(SOURCE ${ARMV8A_BF16_NEONV8_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " -march=armv8.2-a+bf16 -mfpu=neon-fp-armv8 ")
SET_PROPERTY(SOURCE ${ARMV8A_DOTPROD_NEONV8_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " -march=armv8.2-a+dotprod -mfpu=neon-fp-armv8 ")
IF(ANDROID_NDK_MAJOR AND ANDROID_NDK_MAJOR LESS 21)
  SET_PROPERTY(SOURCE ${SOFTFP_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " -mfloat-abi=softfp ")
ENDIF()
IF(ADNN_TARGET_PROCESSOR MATCHES "^arm64" AND NOT MSVC)
  SET_PROPERTY(SOURCE ${ARMV82A_FP16_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " -march=armv8.2-a+fp16 ")
  SET_PROPERTY(SOURCE ${ARMV82A_BF16_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " -march=armv8.2-a+bf16 ")
  SET_PROPERTY(SOURCE ${ARMV82A_DOTPROD_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " -march=armv8.2-a+dotprod ")
  SET_PROPERTY(SOURCE ${ARMV82A_FP16_DOTPROD_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " -march=armv8.2-a+fp16+dotprod ")
ENDIF()
