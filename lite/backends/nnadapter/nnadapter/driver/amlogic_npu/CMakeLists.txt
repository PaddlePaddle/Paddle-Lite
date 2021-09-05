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

set(DEVICE_NAME amlogic_npu)
add_definitions(-DNNADAPTER_DEVICE_NAME=${DEVICE_NAME})

set(DRIVER_NAME ${NNADAPTER_DRIVER_PREFIX}_${DEVICE_NAME})
add_definitions(-DNNADAPTER_DRIVER_NAME=${DRIVER_NAME})

include(dependencies.cmake)

aux_source_directory(./converter CONVERTERS)
aux_source_directory(optimizer OPTIMIZERS)
set(DRIVER_SRCS utility.cc ${OPTIMIZERS} ${CONVERTERS} engine.cc driver.cc)
set(DRIVER_DEPS ${NNADAPTER_UTILITIES} nnadapter_optimizer_symm2asymm ${${DEVICE_NAME}_deps})

add_library(${DRIVER_NAME} SHARED ${DRIVER_SRCS})
target_link_libraries(${DRIVER_NAME} "-Wl,--start-group" ${DRIVER_DEPS} "-Wl,--end-group")
set(NNADAPTER_DRIVERS ${NNADAPTER_DRIVERS} ${DRIVER_NAME} CACHE INTERNAL "")
