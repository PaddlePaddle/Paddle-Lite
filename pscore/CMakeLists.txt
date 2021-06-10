# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.
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
# limitations under the License

cmake_minimum_required(VERSION 3.0)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

set(CMAKE_CXX_STANDARD 14)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT APPLE)
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--as-needed")
endif()

project(pscore CXX C)

find_package(Git REQUIRED)
find_package(Threads REQUIRED)
find_package(CUDA QUIET)


set(THIRD_PARTY_PATH "${CMAKE_BINARY_DIR}/third_party" CACHE STRING
  "A path setting third party libraries download & build directories.")

include(cmake/core.cmake)
include(cmake/external/protobuf.cmake)

include_directories("${CMAKE_SOURCE_DIR}")
include_directories("${CMAKE_BINARY_DIR}")

add_subdirectory(pscore)
