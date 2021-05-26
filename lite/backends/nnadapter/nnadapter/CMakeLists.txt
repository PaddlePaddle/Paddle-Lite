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

# Enable throwing exception when check failed in nnadapter_logging.cc
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fexceptions -fasynchronous-unwind-tables -funwind-tables")

set(NNADAPTER_DRIVER_PREFIX nnadapter_driver)
add_definitions(-DNNADAPTER_DRIVER_PREFIX=${NNADAPTER_DRIVER_PREFIX})

add_library(nnadapter_logging STATIC nnadapter_logging.cc)
add_library(nnadapter_common STATIC nnadapter_common.cc)
add_library(nnadapter_driver STATIC nnadapter_driver.cc)
add_library(nnadapter_optimizer STATIC nnadapter_optimizer.cc)
add_library(nnadapter SHARED nnadapter_api.cc nnadapter_runtime.cc)
target_link_libraries(nnadapter "-Wl,--start-group" nnadapter_logging nnadapter_common nnadapter_driver nnadapter_optimizer "-Wl,--end-group")

add_subdirectory(drivers)
