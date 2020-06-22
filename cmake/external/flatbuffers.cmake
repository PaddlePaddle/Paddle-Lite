# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

INCLUDE(ExternalProject)

# Introduce variables:
# * CMAKE_INSTALL_LIBDIR
INCLUDE(GNUInstallDirs)
SET(LIBDIR "lib")
if(CMAKE_INSTALL_LIBDIR MATCHES ".*lib64$")
  SET(LIBDIR "lib64")
endif()

SET(FLATBUFFERS_SOURCES_DIR ${CMAKE_SOURCE_DIR}/third-party/flatbuffers)
SET(FLATBUFFERS_INSTALL_DIR ${THIRD_PARTY_PATH}/install/flatbuffers)
SET(FLATBUFFERS_INCLUDE_DIR "${FLATBUFFERS_INSTALL_DIR}/include" CACHE PATH "flatbuffers include directory." FORCE)
IF(WIN32)
  set(FLATBUFFERS_LIBRARIES "${FLATBUFFERS_INSTALL_DIR}/${LIBDIR}/libflatbuffers.lib" CACHE FILEPATH "FLATBUFFERS_LIBRARIES" FORCE)
ELSE(WIN32)
  set(FLATBUFFERS_LIBRARIES "${FLATBUFFERS_INSTALL_DIR}/${LIBDIR}/libflatbuffers.a" CACHE FILEPATH "FLATBUFFERS_LIBRARIES" FORCE)
ENDIF(WIN32)

INCLUDE_DIRECTORIES(${FLATBUFFERS_INCLUDE_DIR})

if(NOT HOST_CXX_COMPILER)
  set(HOST_CXX_COMPILER ${CMAKE_CXX_COMPILER})
  set(HOST_C_COMPILER ${CMAKE_C_COMPILER})
endif()

SET(OPTIONAL_ARGS "-DCMAKE_CXX_COMPILER=${HOST_CXX_COMPILER}"
                  "-DCMAKE_C_COMPILER=${HOST_C_COMPILER}")

ExternalProject_Add(
    extern_flatbuffers
    ${EXTERNAL_PROJECT_LOG_ARGS}
    GIT_REPOSITORY  "https://github.com/google/flatbuffers.git"
    GIT_TAG         "v1.12.0"
    SOURCE_DIR      ${FLATBUFFERS_SOURCES_DIR}
    PREFIX          ${FLATBUFFERS_INCLUDE_DIR}
    UPDATE_COMMAND  ""
    CMAKE_ARGS      -DBUILD_STATIC_LIBS=ON
                    -DCMAKE_INSTALL_PREFIX=${FLATBUFFERS_INSTALL_DIR}
                    -DCMAKE_POSITION_INDEPENDENT_CODE=ON
                    -DBUILD_TESTING=OFF
                    -DCMAKE_BUILD_TYPE=${THIRD_PARTY_BUILD_TYPE}
                    ${CROSS_COMPILE_CMAKE_ARGS}
                    ${OPTIONAL_ARGS}
                    ${EXTERNAL_OPTIONAL_ARGS}
    CMAKE_CACHE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=${FLATBUFFERS_INSTALL_DIR}
                     -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
                     -DCMAKE_BUILD_TYPE:STRING=${THIRD_PARTY_BUILD_TYPE}
)
IF(WIN32)
  IF(NOT EXISTS "${FLATBUFFERS_INSTALL_DIR}/${LIBDIR}/libflatbuffers.lib")
    add_custom_command(TARGET extern_flatbuffers POST_BUILD
            COMMAND cmake -E copy ${FLATBUFFERS_INSTALL_DIR}/${LIBDIR}/flatbuffers_static.lib ${FLATBUFFERS_INSTALL_DIR}/${LIBDIR}/libflatbuffers.lib
            )
  ENDIF()
ENDIF(WIN32)
ADD_LIBRARY(flatbuffers STATIC IMPORTED GLOBAL)
SET_PROPERTY(TARGET flatbuffers PROPERTY IMPORTED_LOCATION ${FLATBUFFERS_LIBRARIES})
ADD_DEPENDENCIES(flatbuffers extern_flatbuffers)

SET(FLATBUFFERS_FLATC_EXECUTABLE ${FLATBUFFERS_INSTALL_DIR}/bin/flatc)

function(register_generated_output file_name)
  get_property(tmp GLOBAL PROPERTY FBS_GENERATED_OUTPUTS)
  list(APPEND tmp ${file_name})
  set_property(GLOBAL PROPERTY FBS_GENERATED_OUTPUTS ${tmp})
endfunction(register_generated_output)

function(compile_flatbuffers_schema_to_cpp_opt TARGET SRC_FBS OPT)
  if(FLATBUFFERS_BUILD_LEGACY)
    set(OPT ${OPT};--cpp-std c++0x)
  else()
    # --cpp-std is defined by flatc default settings.
  endif()
  message(STATUS "`${SRC_FBS}`: add generation of C++ code with '${OPT}'")
  get_filename_component(SRC_FBS_DIR ${SRC_FBS} PATH)
  message(STATUS "SRC_FBS_DIR: ${SRC_FBS_DIR}")
  string(REGEX REPLACE "\\.fbs$" "_generated.h" GEN_HEADER ${SRC_FBS})
  add_custom_command(
    OUTPUT ${GEN_HEADER}
    COMMAND "${FLATBUFFERS_FLATC_EXECUTABLE}"
            --cpp --gen-mutable --gen-object-api --reflect-names
            --cpp-ptr-type flatbuffers::unique_ptr # Used to test with C++98 STLs
            ${OPT}
            -I "${CMAKE_CURRENT_SOURCE_DIR}/tests/include_test"
            -o "${SRC_FBS_DIR}"
            "${CMAKE_CURRENT_SOURCE_DIR}/${SRC_FBS}"
    DEPENDS flatbuffers
    COMMENT "Run generation: '${GEN_HEADER}'")
  include_directories(${FLATBUFFERS_INCLUDE_DIR})
  register_generated_output(${GEN_HEADER})
  add_custom_target(${TARGET} ALL DEPENDS ${GEN_HEADER})
endfunction()

set(FRAMEWORK_FBS_DIR "lite/model_parser/flatbuffers")
set(FRAMEWORK_SCHEMA_PATH "${FRAMEWORK_FBS_DIR}/framework.fbs")
compile_flatbuffers_schema_to_cpp_opt(framework_fbs_header ${FRAMEWORK_SCHEMA_PATH} "--no-includes;--gen-compare;--force-empty")
