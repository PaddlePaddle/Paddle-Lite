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

include(ExternalProject)

# Download and extract XTCL archive file
macro(download_and_extract_xtcl_archive_file ARCHIVE_URL DOWNLOAD_DIR INSTALL_DIR)
  get_filename_component(ARCHIVE_FILE_NAME ${ARCHIVE_URL} NAME_WE)
  get_filename_component(ARCHIVE_FILE_EXT ${ARCHIVE_URL} EXT)
  string(REPLACE "-" ";" ARCHIVE_FILE_NAME_TOKEN_LIST ${ARCHIVE_FILE_NAME})
  list(LENGTH ARCHIVE_FILE_NAME_TOKEN_LIST ARCHIVE_FILE_NAME_TOKEN_LIST_LENGTH)
  if(${ARCHIVE_FILE_NAME_TOKEN_LIST_LENGTH} LESS 2)
    message(FATAL_ERROR "Invalid URL is found in the xtcl archive file: ${ARCHIVE_URL}")
  endif()
  list(GET ARCHIVE_FILE_NAME_TOKEN_LIST 0 ARCHIVE_NAME)
  message(STATUS "Downloading and extract ${ARCHIVE_NAME} from ${ARCHIVE_URL} to ${DOWNLOAD_DIR} ...")
  ExternalProject_Add(
    ${ARCHIVE_NAME}
    ${EXTERNAL_PROJECT_LOG_ARGS}
    DOWNLOAD_DIR            ${DOWNLOAD_DIR}
    DOWNLOAD_COMMAND        wget --no-check-certificate -c -q ${ARCHIVE_URL} && tar -xf ${ARCHIVE_FILE_NAME}${ARCHIVE_FILE_EXT}
    CONFIGURE_COMMAND       ""
    BUILD_COMMAND           ""
    UPDATE_COMMAND          ""
    INSTALL_COMMAND         ${CMAKE_COMMAND} -E copy_directory ${DOWNLOAD_DIR}/${ARCHIVE_FILE_NAME} ${INSTALL_DIR}/${ARCHIVE_NAME}
  )
  set(${ARCHIVE_NAME}_root_dir "${INSTALL_DIR}/${ARCHIVE_NAME}" CACHE PATH "${ARCHIVE_NAME} root directory" FORCE)
  set(${ARCHIVE_NAME}_inc_dir "${${ARCHIVE_NAME}_root_dir}/include" CACHE PATH "${ARCHIVE_NAME} include directory" FORCE)
  include_directories(${${ARCHIVE_NAME}_inc_dir})
endmacro()

# Add XTCL lib deps
macro(add_xtcl_lib_deps LIB_PATH)
  set(options "")
  set(oneValueArgs "")
  set(multiValueArgs DEPS)
  cmake_parse_arguments(args "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  get_filename_component(LIB_NAME ${LIB_PATH} NAME_WE)
  get_filename_component(LIB_EXT ${LIB_PATH} EXT)
  string(REPLACE "." "_" LIB_NAME ${LIB_NAME}${LIB_EXT})
  string(REPLACE ";" "_" LIB_DEPS "${args_DEPS}")
  set(LIB_NAME ${LIB_NAME}_${LIB_DEPS})
  set(LIB_TYPE STATIC)
  if(${LIB_EXT} STREQUAL ".so")
    set(LIB_TYPE SHARED)
  endif()
  set(${LIB_NAME}_path ${LIB_PATH} CACHE FILEPATH "${LIB_NAME}_path" FORCE)
  add_library(${LIB_NAME} ${LIB_TYPE} IMPORTED GLOBAL)
  set_property(TARGET ${LIB_NAME} PROPERTY IMPORTED_LOCATION ${${LIB_NAME}_path})
  if(${LIB_TYPE} STREQUAL "SHARED")
    add_custom_target(copy_${LIB_NAME}_lib
      COMMAND cp -r ${${LIB_NAME}_path} ${CMAKE_CURRENT_BINARY_DIR}
      DEPENDS ${args_DEPS}
    )
    add_dependencies(${LIB_NAME} copy_${LIB_NAME}_lib)
  endif()
  set(DEPS ${DEPS} ${LIB_NAME})
endmacro()

if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 5.0)
  message(FATAL_ERROR "Since the Kunlunxin XTCL HAL library needs to be compiled based on C++14, the gcc version is required to be greater than 5.0")
endif()

# Resolve the compilation error caused by the introduction of XTCL and TVM header files
set(CMAKE_CXX_STANDARD 14)
string(REPLACE "-std=c++11" "-std=c++14" CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
string(REPLACE "-Werror" "" CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
string(REPLACE "-Wnon-virtual-dtor" "" CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DDMLC_USE_GLOG=0")
message(STATUS "${CMAKE_CXX_FLAGS}")

if(NOT NNADAPTER_KUNLUNXIN_XTCL_SDK_ROOT)
  set(NNADAPTER_KUNLUNXIN_XTCL_SDK_ROOT $ENV{NNADAPTER_KUNLUNXIN_XTCL_SDK_ROOT})
endif()

if(NOT NNADAPTER_KUNLUNXIN_XTCL_SDK_URL)
  set(NNADAPTER_KUNLUNXIN_XTCL_SDK_URL $ENV{NNADAPTER_KUNLUNXIN_XTCL_SDK_URL})
endif()

if(NOT NNADAPTER_KUNLUNXIN_XTCL_SDK_ENV)
  set(NNADAPTER_KUNLUNXIN_XTCL_SDK_ENV $ENV{NNADAPTER_KUNLUNXIN_XTCL_SDK_ENV})
endif()

if(NNADAPTER_KUNLUNXIN_XTCL_SDK_ROOT)
  message(STATUS "NNADAPTER_KUNLUNXIN_XTCL_SDK_ROOT: ${NNADAPTER_KUNLUNXIN_XTCL_SDK_ROOT}")
  set(XPU_INC_DIR "${NNADAPTER_KUNLUNXIN_XTCL_SDK_ROOT}/XTCL/include" CACHE PATH "XPU include directory" FORCE)
  include_directories(${XPU_INC_DIR})
  if(MSVC)
    add_xtcl_lib_deps(${NNADAPTER_KUNLUNXIN_XTCL_SDK_ROOT}/XTCL/lib/libxpuapi.lib)
    add_xtcl_lib_deps(${NNADAPTER_KUNLUNXIN_XTCL_SDK_ROOT}/XTCL/lib/libxpurt.lib)
    add_xtcl_lib_deps(${NNADAPTER_KUNLUNXIN_XTCL_SDK_ROOT}/XTCL/lib/tvm.lib)
    add_xtcl_lib_deps(${NNADAPTER_KUNLUNXIN_XTCL_SDK_ROOT}/XTCL/lib/xtcl.lib)
  else()
    add_xtcl_lib_deps(${NNADAPTER_KUNLUNXIN_XTCL_SDK_ROOT}/XTCL/shlib/libxpuapi.so)
    add_xtcl_lib_deps(${NNADAPTER_KUNLUNXIN_XTCL_SDK_ROOT}/XTCL/runtime/shlib/libxpurt.so)
    add_xtcl_lib_deps(${NNADAPTER_KUNLUNXIN_XTCL_SDK_ROOT}/XTCL/shlib/libxpujitc.so)
    add_xtcl_lib_deps(${NNADAPTER_KUNLUNXIN_XTCL_SDK_ROOT}/XTCL/shlib/libLLVM-10.so)
    add_xtcl_lib_deps(${NNADAPTER_KUNLUNXIN_XTCL_SDK_ROOT}/XTCL/shlib/libtvm.so)
    add_custom_target(copy_bkcl_lib
      COMMAND cp -r ${NNADAPTER_KUNLUNXIN_XTCL_SDK_ROOT}/XTCL/shlib/libbkcl.so ${CMAKE_CURRENT_BINARY_DIR}
    )
    add_xtcl_lib_deps(${NNADAPTER_KUNLUNXIN_XTCL_SDK_ROOT}/XTCL/shlib/libxtcl.so DEPS copy_bkcl_lib)
  endif()
else()
  if(NOT NNADAPTER_KUNLUNXIN_XTCL_SDK_URL)
    set(NNADAPTER_KUNLUNXIN_XTCL_SDK_URL "http://baidu-kunlun-product.su.bcebos.com/KL-SDK/klsdk-dev_paddle")
  endif()
  if(NOT NNADAPTER_KUNLUNXIN_XTCL_SDK_ENV)
    # Detect NNADAPTER_KUNLUNXIN_XTCL_SDK_ENV automatically
    if(WIN32)
      set(HOST_SYSTEM_NAME "win32")
    else()
      if(APPLE)
        set(HOST_SYSTEM_NAME "macosx")
        exec_program(sw_vers ARGS -productVersion OUTPUT_VARIABLE HOST_SYSTEM_VERSION)
        string(REGEX MATCH "[0-9]+.[0-9]+" MACOS_VERSION "${HOST_SYSTEM_VERSION}")
        if(NOT DEFINED $ENV{MACOSX_DEPLOYMENT_TARGET})
          # Set cache variable - end user may change this during ccmake or cmake-gui configure.
          set(CMAKE_OSX_DEPLOYMENT_TARGET ${MACOS_VERSION} CACHE STRING "Minimum OS X version to target for deployment (at runtime); newer APIs weak linked. Set to empty string for default value.")
        endif()
      else()
        if(EXISTS "/etc/issue")
          file(READ "/etc/issue" LINUX_ISSUE)
          if(LINUX_ISSUE MATCHES "CentOS")
            set(HOST_SYSTEM_NAME "centos")
          elseif(LINUX_ISSUE MATCHES "Debian")
            set(HOST_SYSTEM_NAME "debian")
          elseif(LINUX_ISSUE MATCHES "Ubuntu")
            set(HOST_SYSTEM_NAME "ubuntu")
          elseif(LINUX_ISSUE MATCHES "Red Hat")
            set(HOST_SYSTEM_NAME "redhat")
          elseif(LINUX_ISSUE MATCHES "Fedora")
            set(HOST_SYSTEM_NAME "fedora")
          endif()
          string(REGEX MATCH "(([0-9]+)\\.)+([0-9]+)" HOST_SYSTEM_VERSION "${LINUX_ISSUE}")
        endif()
        if(EXISTS "/etc/redhat-release")
          file(READ "/etc/redhat-release" LINUX_ISSUE)
          if(LINUX_ISSUE MATCHES "CentOS")
            set(HOST_SYSTEM_NAME "centos")
          endif()
        endif()
        if(NOT HOST_SYSTEM_NAME)
          set(HOST_SYSTEM_NAME ${CMAKE_SYSTEM_NAME})
        endif()
      endif()
    endif()
    if(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
      set(NNADAPTER_KUNLUNXIN_XTCL_SDK_ENV "kylin_aarch64")
    elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64" OR CMAKE_SYSTEM_PROCESSOR STREQUAL "amd64")
      set(NNADAPTER_KUNLUNXIN_XTCL_SDK_ENV "bdcentos_x86_64")
      if(${HOST_SYSTEM_NAME} STREQUAL "ubuntu")
        set(NNADAPTER_KUNLUNXIN_XTCL_SDK_ENV "ubuntu_x86_64")
      endif()
    else()
      message(FATAL_ERROR "${CMAKE_SYSTEM_PROCESSOR} isn't supported by Kunlunxin XTCL SDK.")
      endif()
    if(NOT NNADAPTER_KUNLUNXIN_XTCL_SDK_ENV)
      message(FATAL_ERROR "Unable to get the NNADAPTER_KUNLUNXIN_XTCL_SDK_ENV of the current environment automatically, please set NNADAPTER_KUNLUNXIN_XTCL_SDK_ENV manually")
    endif()
  endif()
  
  set(NNADAPTER_KUNLUNXIN_XTCL_SDK_DOWNLOAD_DIR ${THIRD_PARTY_PATH}/xtcl)
  set(NNADAPTER_KUNLUNXIN_XTCL_SDK_INSTALL_DIR ${THIRD_PARTY_PATH}/install/xtcl)
  download_and_extract_xtcl_archive_file(${NNADAPTER_KUNLUNXIN_XTCL_SDK_URL}/xdnn-${NNADAPTER_KUNLUNXIN_XTCL_SDK_ENV}.tar.gz ${NNADAPTER_KUNLUNXIN_XTCL_SDK_DOWNLOAD_DIR} ${NNADAPTER_KUNLUNXIN_XTCL_SDK_INSTALL_DIR})
  download_and_extract_xtcl_archive_file(${NNADAPTER_KUNLUNXIN_XTCL_SDK_URL}/xre-${NNADAPTER_KUNLUNXIN_XTCL_SDK_ENV}.tar.gz ${NNADAPTER_KUNLUNXIN_XTCL_SDK_DOWNLOAD_DIR} ${NNADAPTER_KUNLUNXIN_XTCL_SDK_INSTALL_DIR})
  download_and_extract_xtcl_archive_file(${NNADAPTER_KUNLUNXIN_XTCL_SDK_URL}/xtdk-${NNADAPTER_KUNLUNXIN_XTCL_SDK_ENV}.tar.gz ${NNADAPTER_KUNLUNXIN_XTCL_SDK_DOWNLOAD_DIR} ${NNADAPTER_KUNLUNXIN_XTCL_SDK_INSTALL_DIR})
  download_and_extract_xtcl_archive_file(${NNADAPTER_KUNLUNXIN_XTCL_SDK_URL}/xccl-${NNADAPTER_KUNLUNXIN_XTCL_SDK_ENV}.tar.gz ${NNADAPTER_KUNLUNXIN_XTCL_SDK_DOWNLOAD_DIR} ${NNADAPTER_KUNLUNXIN_XTCL_SDK_INSTALL_DIR})
  download_and_extract_xtcl_archive_file(${NNADAPTER_KUNLUNXIN_XTCL_SDK_URL}/xtcl-${NNADAPTER_KUNLUNXIN_XTCL_SDK_ENV}.tar.gz ${NNADAPTER_KUNLUNXIN_XTCL_SDK_DOWNLOAD_DIR} ${NNADAPTER_KUNLUNXIN_XTCL_SDK_INSTALL_DIR})
  if(MSVC)
    # TODO(hong19860320) Supports for Visual Studio and Win32
    message(FATAL_ERROR "Unable to use NNADAPTER_KUNLUNXIN_XTCL_SDK_URL and NNADAPTER_KUNLUNXIN_XTCL_SDK_ENV to download the XTCL SDK to compile the library based on visual studio on windows, please use NNADAPTER_KUNLUNXIN_XTCL_SDK_ROOT to specify the root directory of XPU Toolchain directly.")
  else()
    add_xtcl_lib_deps(${NNADAPTER_KUNLUNXIN_XTCL_SDK_INSTALL_DIR}/xdnn/so/libxpuapi.so DEPS xdnn)
    add_xtcl_lib_deps(${NNADAPTER_KUNLUNXIN_XTCL_SDK_INSTALL_DIR}/xre/so/libxpurt.so DEPS xre)
    add_xtcl_lib_deps(${NNADAPTER_KUNLUNXIN_XTCL_SDK_INSTALL_DIR}/xtdk/so/libxpujitc.so DEPS xtdk)
    add_xtcl_lib_deps(${NNADAPTER_KUNLUNXIN_XTCL_SDK_INSTALL_DIR}/xtdk/so/libLLVM-10.so DEPS xtdk)
    add_xtcl_lib_deps(${NNADAPTER_KUNLUNXIN_XTCL_SDK_INSTALL_DIR}/xtcl/so/libtvm.so DEPS xtcl)
    add_custom_target(copy_bkcl_xccl_lib
      COMMAND cp -r ${NNADAPTER_KUNLUNXIN_XTCL_SDK_INSTALL_DIR}/xccl/so/libbkcl.so ${CMAKE_CURRENT_BINARY_DIR}
      DEPENDS xccl
    )
    add_xtcl_lib_deps(${NNADAPTER_KUNLUNXIN_XTCL_SDK_INSTALL_DIR}/xtcl/so/libxtcl.so DEPS xtcl copy_bkcl_xccl_lib)
  endif()
endif()
