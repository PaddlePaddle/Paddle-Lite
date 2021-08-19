# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

if(NOT WIN32 OR NOT MSVC)
    return()
endif()

include(ExternalProject)

set(DIRENT_PROJECT "extern_dirent")
set(DIRENT_PREFIX_DIR    ${THIRD_PARTY_PATH}/dirent)
set(DIRENT_INSTALL_ROOT  ${THIRD_PARTY_PATH}/install/dirent)
set(DIRENT_INC_DIR       ${DIRENT_INSTALL_ROOT}/include)
set(DIRENT_DOWNLOAD_DIR    ${DIRENT_PREFIX_DIR}/src/${DIRENT_PROJECT})

include_directories(${DIRENT_INC_DIR})

set(DIRENT_URL "http://paddle-inference-dist.bj.bcebos.com/PaddleLite_ThirdParty%2Fdirent-1.23.2.zip")
ExternalProject_Add(
    dirent_header
    ${EXTERNAL_PROJECT_LOG_ARGS}
    DEPENDS             ""
    GIT_TAG             "v1.23.2"  # 9 May 2018
    URL                 ${DIRENT_URL}
    DOWNLOAD_DIR        ${DIRENT_DOWNLOAD_DIR}
    DOWNLOAD_NAME   "dirent-1.23.2.zip"
    DOWNLOAD_NO_PROGRESS 1
    PREFIX              ${DIRENT_PREFIX_DIR}
    UPDATE_COMMAND      ""
    INSTALL_COMMAND
        ${CMAKE_COMMAND} -E copy_directory ${DIRENT_DOWNLOAD_DIR}/include ${DIRENT_INC_DIR}
)
