# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

include(ExternalProject)

set(NNADAPTER_EXTERN_PROTOBUF_PROJECT nnadapter_extern_protobuf)
set(NNADAPTER_EXTERN_PROTOBUF_SOURCES_DIR ${THIRD_PARTY_PATH}/protobuf)
set(NNADAPTER_EXTERN_PROTOBUF_INSTALL_DIR ${THIRD_PARTY_PATH}/install/protobuf)
set(NNADAPTER_EXTERN_PROTOBUF_BUILD_COMMAND $(MAKE) -j)

ExternalProject_Add(
  ${NNADAPTER_EXTERN_PROTOBUF_PROJECT}
  ${EXTERNAL_PROJECT_LOG_ARGS}
  GIT_REPOSITORY      "https://github.com/protocolbuffers/protobuf.git"
  GIT_TAG             "v3.16.3"
  GIT_CONFIG          user.name=anonymous user.email=anonymous@anonymous.com
  SOURCE_DIR          ${NNADAPTER_EXTERN_PROTOBUF_SOURCES_DIR}
  PREFIX              ${NNADAPTER_EXTERN_PROTOBUF_INSTALL_DIR}
  PATCH_COMMAND       ""
  BUILD_COMMAND       ${NNADAPTER_EXTERN_PROTOBUF_BUILD_COMMAND}
  CMAKE_ARGS          -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
                      -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
                      -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
                      -DCMAKE_INSTALL_PREFIX=${NNADAPTER_EXTERN_PROTOBUF_INSTALL_DIR}
                      ${CROSS_COMPILE_CMAKE_ARGS}
)

include_directories(${NNADAPTER_EXTERN_PROTOBUF_INSTALL_DIR}/include)
