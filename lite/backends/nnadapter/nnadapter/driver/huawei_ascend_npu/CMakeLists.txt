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

string(REGEX MATCHALL "[0-9]+" CANN_VERSION_ALL_LIST "${NNADAPTER_HUAWEI_ASCEND_NPU_SDK_VERSION}")
MESSAGE("CANN_VERSION_ALL_LIST: ${CANN_VERSION_ALL_LIST}")
list(LENGTH CANN_VERSION_ALL_LIST CANN_VERSION_ALL_LIST_LENGTH)
MESSAGE("CANN_VERSION_ALL_LIST_LENGTH: ${CANN_VERSION_ALL_LIST_LENGTH}")
if(CANN_VERSION_ALL_LIST_LENGTH MATCHES "3")
	list(GET CANN_VERSION_ALL_LIST 0 CANN_MAJOR_VERSION)
	list(GET CANN_VERSION_ALL_LIST 1 CANN_MINOR_VERSION)
	list(GET CANN_VERSION_ALL_LIST 2 CANN_PATCH_VERSION)
	set(CANN_BUGFIX_VERSION 0)
elseif(CANN_VERSION_ALL_LIST_LENGTH MATCHES "4")
	list(GET CANN_VERSION_ALL_LIST 0 CANN_MAJOR_VERSION)
	list(GET CANN_VERSION_ALL_LIST 1 CANN_MINOR_VERSION)
	list(GET CANN_VERSION_ALL_LIST 2 CANN_PATCH_VERSION)
	list(GET CANN_VERSION_ALL_LIST 3 CANN_BUGFIX_VERSION)
else()
    message(FATAL_ERROR "The Huawei Ascend NPU CANN version is wrong: ${NNADAPTER_HUAWEI_ASCEND_NPU_SDK_VERSION}")
endif()
message("NNADAPTER_HUAWEI_ASCEND_NPU_CANN_MAJOR_VERSION: ${CANN_MAJOR_VERSION}")
message("NNADAPTER_HUAWEI_ASCEND_NPU_CANN_MINOR_VERSION: ${CANN_MINOR_VERSION}")
message("NNADAPTER_HUAWEI_ASCEND_NPU_CANN_PATCH_VERSION: ${CANN_PATCH_VERSION}")
message("NNADAPTER_HUAWEI_ASCEND_NPU_CANN_BUGFIX_VERSION: ${CANN_BUGFIX_VERSION}")
math(EXPR NNADAPTE_HUAWEI_ASCEND_NPU_CANN_VERSION
"${CANN_MAJOR_VERSION} * 1000 +
 ${CANN_MINOR_VERSION} * 100 + ${CANN_PATCH_VERSION} * 10 + ${CANN_BUGFIX_VERSION}")
add_definitions(-DNNADAPTE_HUAWEI_ASCEND_NPU_CANN_VERSION=${NNADAPTE_HUAWEI_ASCEND_NPU_CANN_VERSION})

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
