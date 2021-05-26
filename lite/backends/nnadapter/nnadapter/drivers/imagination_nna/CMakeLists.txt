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

set(NNADAPTER_DRIVER_NAME imagination_nna)
add_definitions(-DNNADAPTER_DRIVER_NAME=${NNADAPTER_DRIVER_NAME})

set(NNADAPTER_DRIVER_TARGET ${NNADAPTER_DRIVER_PREFIX}_${NNADAPTER_DRIVER_NAME})
add_definitions(-DNNADAPTER_DRIVER_TARGET=${NNADAPTER_DRIVER_TARGET})

set(NNADAPTER_DRIVER_SRCS driver.cc)
set(NNADAPTER_DRIVER_DEPS nnadapter_common nnadapter_logging nnadapter_driver ${${NNADAPTER_DRIVER_NAME}_deps})

add_library(${NNADAPTER_DRIVER_TARGET} SHARED ${NNADAPTER_DRIVER_SRCS})
target_link_libraries(${NNADAPTER_DRIVER_TARGET} "-Wl,--start-group" ${NNADAPTER_DRIVER_DEPS} "-Wl,--end-group")
set(nnadapter_drivers ${nnadapter_drivers} ${NNADAPTER_DRIVER_TARGET} CACHE INTERNAL "")
