# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

option(MSVC_STATIC_CRT "use static C Runtime library by default" ON)

set(CMAKE_SUPPRESS_REGENERATION ON)
set(CMAKE_STATIC_LIBRARY_PREFIX lib)

# Toolchain
if(MSVC_STATIC_CRT)
  set(CMAKE_C_FLAGS_DEBUG   "${CMAKE_C_FLAGS_DEBUG} /bigobj /MTd")
  set(CMAKE_C_FLAGS_RELEASE  "${CMAKE_C_FLAGS_RELEASE} /bigobj /MT")
  set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG} /bigobj /MTd")
  set(CMAKE_CXX_FLAGS_RELEASE   "${CMAKE_CXX_FLAGS_RELEASE} /bigobj /MT")
else(MSVC_STATIC_CRT)
  set(CMAKE_C_FLAGS_DEBUG   "${CMAKE_C_FLAGS_DEBUG} /bigobj /MDd")
  set(CMAKE_C_FLAGS_RELEASE  "${CMAKE_C_FLAGS_RELEASE} /bigobj /MD")
  set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG} /bigobj /MDd")
  set(CMAKE_CXX_FLAGS_RELEASE   "${CMAKE_CXX_FLAGS_RELEASE} /bigobj /MD")
endif(MSVC_STATIC_CRT)
add_compile_options(/wd4068 /wd4129 /wd4244 /wd4267 /wd4297 /wd4530 /wd4577 /wd4819 /wd4838)
add_compile_options(/MP)
message(STATUS "Using parallel compiling (/MP)")
set(PADDLE_LINK_FLAGS "/IGNORE:4006 /IGNORE:4098 /IGNORE:4217 /IGNORE:4221")
set(CMAKE_STATIC_LINKER_FLAGS  "${CMAKE_STATIC_LINKER_FLAGS} ${PADDLE_LINK_FLAGS}")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${PADDLE_LINK_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} ${PADDLE_LINK_FLAGS}")
if(MSVC)
  include(external/dirent)
endif()

# Definitions
add_definitions("/DGOOGLE_GLOG_DLL_DECL=")
add_definitions(-DGLOG_NO_ABBREVIATED_SEVERITIES)
add_definitions(-DNOMINMAX)
