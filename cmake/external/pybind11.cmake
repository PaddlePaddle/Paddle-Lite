# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.
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

if(NOT LITE_WITH_PYTHON)
    return()
endif()

include(ExternalProject)

set(PYBIND_SOURCE_DIR ${PADDLE_SOURCE_DIR}/third-party/pybind)

include_directories(${PYBIND_SOURCE_DIR}/include)

ExternalProject_Add(
        extern_pybind
        ${EXTERNAL_PROJECT_LOG_ARGS}
        GIT_REPOSITORY  ""
        GIT_TAG         8de7772cc72daca8e947b79b83fea46214931604
        SOURCE_DIR      ${PYBIND_SOURCE_DIR}
        UPDATE_COMMAND  ""
        CONFIGURE_COMMAND ""
        BUILD_COMMAND     ""
        INSTALL_COMMAND   ""
        TEST_COMMAND      ""
)

if(${CMAKE_VERSION} VERSION_LESS "3.3.0")
    set(dummyfile ${CMAKE_CURRENT_BINARY_DIR}/pybind_dummy.c)
    file(WRITE ${dummyfile} "const char * dummy_pybind = \"${dummyfile}\";")
    add_library(pybind STATIC ${dummyfile})
else()
    add_library(pybind INTERFACE)
endif()

add_dependencies(pybind extern_pybind)
