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

set(NNADAPTER_DEVICE_NAME huawei_kirin_npu)
add_definitions(-DNNADAPTER_DEVICE_NAME=${NNADAPTER_DEVICE_NAME})

set(NNADAPTER_DRIVER_NAME ${NNADAPTER_DRIVER_PREFIX}_${NNADAPTER_DEVICE_NAME})
add_definitions(-DNNADAPTER_DRIVER_NAME=${NNADAPTER_DRIVER_NAME})

include(dependencies.cmake)

aux_source_directory(./converter converter)
set(NNADAPTER_DRIVER_SRCS utility.cc converter.cc ${converter} driver.cc)
set(NNADAPTER_DRIVER_DEPS ${nnadapter_utility} ${${NNADAPTER_DEVICE_NAME}_deps})

add_library(${NNADAPTER_DRIVER_NAME} SHARED ${NNADAPTER_DRIVER_SRCS})
target_link_libraries(${NNADAPTER_DRIVER_NAME} "-Wl,--start-group" ${NNADAPTER_DRIVER_DEPS} "-Wl,--end-group")
set(nnadapter_driver ${nnadapter_driver} ${NNADAPTER_DRIVER_NAME} CACHE INTERNAL "")
