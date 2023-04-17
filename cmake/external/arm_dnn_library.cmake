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

IF(NOT LITE_WITH_ARM)
 RETURN()
ENDIF()

INCLUDE(ExternalProject)

SET(ARM_DNN_LIBRARY_PROJECT "extern_arm_dnn_library")
SET(ARM_DNN_LIBRARY_SOURCE_DIR ${PADDLE_SOURCE_DIR}/lite/backends/arm/arm_dnn_library)
SET(ARM_DNN_LIBRARY_INSTALL_DIR ${THIRD_PARTY_PATH}/install/arm_dnn_library)
IF(WIN32)
  SET(ARM_DNN_LIBRARY_BINARY_PATH "${ARM_DNN_LIBRARY_INSTALL_DIR}/lib/libarm_dnn_library.lib" CACHE FILEPATH "Arm DNN library." FORCE)
ELSE()
  SET(ARM_DNN_LIBRARY_BINARY_PATH "${ARM_DNN_LIBRARY_INSTALL_DIR}/lib/libarm_dnn_library.a" CACHE FILEPATH "Arm DNN library." FORCE)
ENDIF()

INCLUDE_DIRECTORIES(${ARM_DNN_LIBRARY_SOURCE_DIR}/include)

ExternalProject_Add(
  ${ARM_DNN_LIBRARY_PROJECT}
  SOURCE_DIR        ${ARM_DNN_LIBRARY_SOURCE_DIR}
  BUILD_ALWAYS      1
  PREFIX            ${ARM_DNN_LIBRARY_INSTALL_DIR}
  CMAKE_ARGS        ${CROSS_COMPILE_CMAKE_ARGS}
                    -DARM_DNN_LIBRARY_LIBRARY_TYPE=static
                    -DCMAKE_INSTALL_PREFIX=${ARM_DNN_LIBRARY_INSTALL_DIR}
                    -DCMAKE_POSITION_INDEPENDENT_CODE=ON
                    -DCMAKE_BUILD_TYPE=${THIRD_PARTY_BUILD_TYPE}
  CMAKE_CACHE_ARGS  -DCMAKE_INSTALL_PREFIX=${ARM_DNN_LIBRARY_INSTALL_DIR}
                    -DCMAKE_POSITION_INDEPENDENT_CODE=ON
                    -DCMAKE_BUILD_TYPE=${THIRD_PARTY_BUILD_TYPE}
)

ADD_LIBRARY(arm_dnn_library STATIC IMPORTED GLOBAL)
SET_PROPERTY(TARGET arm_dnn_library PROPERTY IMPORTED_LOCATION ${ARM_DNN_LIBRARY_BINARY_PATH})
ADD_DEPENDENCIES(arm_dnn_library ${ARM_DNN_LIBRARY_PROJECT})
