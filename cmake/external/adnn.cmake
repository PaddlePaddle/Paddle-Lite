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

SET(ADNN_PROJECT "extern_adnn")
SET(ADNN_SOURCE_DIR ${PADDLE_SOURCE_DIR}/lite/backends/arm/adnn)
SET(ADNN_INSTALL_DIR ${THIRD_PARTY_PATH}/install/adnn)
IF(WIN32)
  SET(ADNN_LIBRARY_PATH "${ADNN_INSTALL_DIR}/lib/libadnn.lib" CACHE FILEPATH "Arm DNN library." FORCE)
ELSE(WIN32)
  SET(ADNN_LIBRARY_PATH "${ADNN_INSTALL_DIR}/lib/libadnn.a" CACHE FILEPATH "Arm DNN library." FORCE)
ENDIF(WIN32)

INCLUDE_DIRECTORIES(${ADNN_SOURCE_DIR}/include)

ExternalProject_Add(
  ${ADNN_PROJECT}
  SOURCE_DIR        ${ADNN_SOURCE_DIR}
  PREFIX            ${ADNN_INSTALL_DIR}
  CMAKE_ARGS        ${CROSS_COMPILE_CMAKE_ARGS}
                    -DADNN_LIBRARY_TYPE=static
                    -DCMAKE_INSTALL_PREFIX=${ADNN_INSTALL_DIR}
                    -DCMAKE_POSITION_INDEPENDENT_CODE=ON
                    -DCMAKE_BUILD_TYPE=${THIRD_PARTY_BUILD_TYPE}
  CMAKE_CACHE_ARGS  -DCMAKE_INSTALL_PREFIX=${ADNN_INSTALL_DIR}
                    -DCMAKE_POSITION_INDEPENDENT_CODE=ON
                    -DCMAKE_BUILD_TYPE=${THIRD_PARTY_BUILD_TYPE}
)

ADD_LIBRARY(adnn STATIC IMPORTED GLOBAL)
SET_PROPERTY(TARGET adnn PROPERTY IMPORTED_LOCATION ${ADNN_LIBRARY_PATH})
ADD_DEPENDENCIES(adnn ${ADNN_PROJECT})
