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

set(DEVICE_NAME huawei_ascend_npu)
add_definitions(-DNNADAPTER_DEVICE_NAME=${DEVICE_NAME})
add_definitions(-DNNADAPTER_DEVICE_SYMBOL=${NNADAPTER_DEVICE_SYMBOL_PREFIX}${DEVICE_NAME})

include(dependencies.cmake)

aux_source_directory(converter CONVERTERS)
aux_source_directory(optimizer OPTIMIZERS)
set(SRCS utility.cc ${OPTIMIZERS} ${CONVERTERS} model_client.cc engine.cc driver.cc)
set(DEPS ${NNADAPTER_CORE}
        ${NNADAPTER_UTILITIES}
        ${${DEVICE_NAME}_deps}
        nnadapter_optimizer_fuse_matmul_add_into_fully_connected)

add_library(${DEVICE_NAME} SHARED ${SRCS})
target_link_libraries(${DEVICE_NAME} "-Wl,--start-group" ${DEPS} "-Wl,--end-group")
set(NNADAPTER_DEVICES ${NNADAPTER_DEVICES} ${DEVICE_NAME} CACHE INTERNAL "")
