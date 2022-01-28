# Copyright (c) 2017 PaddlePaddle Authors. All Rights Reserved.
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

IF(NOT ${WITH_MKLML})
  return()
ENDIF(NOT ${WITH_MKLML})

INCLUDE(ExternalProject)
SET(MKLML_DST_DIR       "mklml")
SET(MKLML_INSTALL_ROOT  "${THIRD_PARTY_PATH}/install")
SET(MKLML_INSTALL_DIR   ${MKLML_INSTALL_ROOT}/${MKLML_DST_DIR})
SET(MKLML_ROOT          ${MKLML_INSTALL_DIR})
SET(MKLML_INC_DIR       ${MKLML_ROOT}/include)
SET(MKLML_LIB_DIR       ${MKLML_ROOT}/lib)
SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_RPATH}" "${MKLML_ROOT}/lib")
SET(MKLML_MD5 "")

IF(WITH_STATIC_MKL)
    SET(TIME_VERSION "2019.1.144")
    IF(WIN32)
      IF(MSVC)
        IF("${CMAKE_GENERATOR_PLATFORM}" STREQUAL "x64")
          SET(MKLML_PLAT "windows_x64_static")
          SET(MKLML_VER "mklml_win_${TIME_VERSION}_static_x64" CACHE STRING "" FORCE)
          SET(MKLML_URL "https://paddlelite-data.bj.bcebos.com/third_party_libs/${MKLML_VER}.zip" CACHE STRING "" FORCE)
          SET(MKLML_MD5 "MD5=eacd0357a52b718f5175e98e7fdcf0a4")
          SET(MKLML_LP64_LIB            ${MKLML_LIB_DIR}/mkl_intel_lp64.lib)
          SET(MKLML_THREAD_LIB          ${MKLML_LIB_DIR}/mkl_intel_thread.lib)
          SET(MKLML_CORE_LIB            ${MKLML_LIB_DIR}/mkl_core.lib)
          SET(MKLML_IOMP_LIB            ${MKLML_LIB_DIR}/libiomp5md.lib)
          SET(MKLML_SHARED_IOMP_LIB     ${MKLML_LIB_DIR}/libiomp5md.dll)
        ELSE()
          SET(MKLML_PLAT "windows_x86_static")
          SET(MKLML_VER "mklml_win_${TIME_VERSION}_static_x86" CACHE STRING "" FORCE)
          SET(MKLML_URL "https://paddlelite-data.bj.bcebos.com/third_party_libs/${MKLML_VER}.zip" CACHE STRING "" FORCE)
          SET(MKLML_MD5 "MD5=bad6696e6a24a1be1b33a7f23733e9f5")
          SET(MKLML_LP64_LIB            ${MKLML_LIB_DIR}/mkl_intel_c.lib)
          SET(MKLML_THREAD_LIB          ${MKLML_LIB_DIR}/mkl_intel_thread.lib)
          SET(MKLML_CORE_LIB            ${MKLML_LIB_DIR}/mkl_core.lib)
          SET(MKLML_IOMP_LIB            ${MKLML_LIB_DIR}/libiomp5md.lib)
          SET(MKLML_SHARED_IOMP_LIB     ${MKLML_LIB_DIR}/libiomp5md.dll)
        ENDIF()
      ELSE()
        #  Ninja do not support CMAKE_GENERATOR_PLATFORM.
        IF("${ARCH}" STREQUAL "amd64")
          SET(MKLML_PLAT "windows_x64_static")
          SET(MKLML_VER "mklml_win_${TIME_VERSION}_static_x64" CACHE STRING "" FORCE)
          SET(MKLML_URL "https://paddlelite-data.bj.bcebos.com/third_party_libs/${MKLML_VER}.zip" CACHE STRING "" FORCE)
          SET(MKLML_MD5 "MD5=eacd0357a52b718f5175e98e7fdcf0a4")
          SET(MKLML_LP64_LIB            ${MKLML_LIB_DIR}/mkl_intel_lp64.lib)
          SET(MKLML_THREAD_LIB          ${MKLML_LIB_DIR}/mkl_intel_thread.lib)
          SET(MKLML_CORE_LIB            ${MKLML_LIB_DIR}/mkl_core.lib)
          SET(MKLML_IOMP_LIB            ${MKLML_LIB_DIR}/libiomp5md.lib)
          SET(MKLML_SHARED_IOMP_LIB     ${MKLML_LIB_DIR}/libiomp5md.dll)
        ELSE()
          SET(MKLML_PLAT "windows_x86_static")
          SET(MKLML_VER "mklml_win_${TIME_VERSION}_static_x86" CACHE STRING "" FORCE)
          SET(MKLML_URL "https://paddlelite-data.bj.bcebos.com/third_party_libs/${MKLML_VER}.zip" CACHE STRING "" FORCE)
          SET(MKLML_MD5 "MD5=bad6696e6a24a1be1b33a7f23733e9f5")
          SET(MKLML_LP64_LIB            ${MKLML_LIB_DIR}/mkl_intel_c.lib)
          SET(MKLML_THREAD_LIB          ${MKLML_LIB_DIR}/mkl_intel_thread.lib)
          SET(MKLML_CORE_LIB            ${MKLML_LIB_DIR}/mkl_core.lib)
          SET(MKLML_IOMP_LIB            ${MKLML_LIB_DIR}/libiomp5md.lib)
          SET(MKLML_SHARED_IOMP_LIB     ${MKLML_LIB_DIR}/libiomp5md.dll)
        ENDIF()
      ENDIF()
    ELSEIF(APPLE)
        #TODO(intel-huying):
        #  Now enable Erf function in mklml library temporarily, it will be updated as offical version later.
        SET(MKLML_PLAT "apple_static")
        SET(MKLML_VER "mklml_mac_${TIME_VERSION}_static" CACHE STRING "" FORCE)
        SET(MKLML_URL "https://paddlelite-data.bj.bcebos.com/third_party_libs/${MKLML_VER}.tgz" CACHE STRING "" FORCE)
        SET(MKLML_MD5 "MD5=9776e2e3fe27e5232d0356dd6d6309b6")
        SET(MKLML_LP64_LIB            ${MKLML_LIB_DIR}/libmkl_intel_lp64.a)
        SET(MKLML_THREAD_LIB          ${MKLML_LIB_DIR}/libmkl_intel_thread.a)
        SET(MKLML_CORE_LIB            ${MKLML_LIB_DIR}/libmkl_core.a)
        SET(MKLML_IOMP_LIB            ${MKLML_LIB_DIR}/libiomp5.dylib)
        SET(MKLML_SHARED_IOMP_LIB     ${MKLML_LIB_DIR}/libiomp5.dylib)
    ELSE()
        #TODO(intel-huying):
        #  Now enable Erf function in mklml library temporarily, it will be updated as offical version later.
        SET(MKLML_PLAT "linux_static")
        SET(MKLML_VER "mklml_lnx_${TIME_VERSION}_static" CACHE STRING "" FORCE)
        SET(MKLML_URL "https://paddlelite-data.bj.bcebos.com/third_party_libs/${MKLML_VER}.tgz" CACHE STRING "" FORCE)
        SET(MKLML_MD5 "MD5=02b9d6bdad657e24c253cd2f094e8533")
        SET(MKLML_LP64_LIB            ${MKLML_LIB_DIR}/libmkl_intel_lp64.a)
        SET(MKLML_THREAD_LIB          ${MKLML_LIB_DIR}/libmkl_intel_thread.a)
        SET(MKLML_CORE_LIB            ${MKLML_LIB_DIR}/libmkl_core.a)
        SET(MKLML_IOMP_LIB            ${MKLML_LIB_DIR}/libiomp5.so)
        SET(MKLML_SHARED_IOMP_LIB     ${MKLML_LIB_DIR}/libiomp5.so)
    ENDIF()
ELSE()
    SET(TIME_VERSION "2019.0.1.20181227")
    IF(WIN32)
        SET(MKLML_PLAT "windows_shared")
        SET(MKLML_VER "mklml_win_${TIME_VERSION}" CACHE STRING "" FORCE)
        SET(MKLML_URL "https://paddlepaddledeps.bj.bcebos.com/${MKLML_VER}.zip" CACHE STRING "" FORCE)
        SET(MKLML_MD5 "MD5=6368118f678dbf505e42225860bcff67")
        SET(MKLML_LIB                 ${MKLML_LIB_DIR}/mklml.lib)
        SET(MKLML_IOMP_LIB            ${MKLML_LIB_DIR}/libiomp5md.lib)
        SET(MKLML_SHARED_LIB          ${MKLML_LIB_DIR}/mklml.dll)
        SET(MKLML_SHARED_LIB_DEPS     ${MKLML_LIB_DIR}/msvcr120.dll)
        SET(MKLML_SHARED_IOMP_LIB     ${MKLML_LIB_DIR}/libiomp5md.dll)
    ELSEIF(APPLE)
        #TODO(intel-huying):
        #  Now enable Erf function in mklml library temporarily, it will be updated as offical version later.
        SET(MKLML_PLAT "apple_shared")
        SET(MKLML_VER "mklml_mac_2019.0.5.20190502" CACHE STRING "" FORCE)
        SET(MKLML_URL "https://paddlelite-data.bj.bcebos.com/third_party_libs/${MKLML_VER}.tgz" CACHE STRING "" FORCE)
        SET(MKLML_MD5 "MD5=0a3d83ec1fed9ea318e8573bb5e14c24")
        SET(MKLML_LIB                 ${MKLML_LIB_DIR}/libmklml.dylib)
        SET(MKLML_IOMP_LIB            ${MKLML_LIB_DIR}/libiomp5.dylib)
        SET(MKLML_SHARED_LIB          ${MKLML_LIB_DIR}/libmklml.dylib)
        SET(MKLML_SHARED_IOMP_LIB     ${MKLML_LIB_DIR}/libiomp5.dylib)
    ELSE()
        #TODO(intel-huying):
        #  Now enable Erf function in mklml library temporarily, it will be updated as offical version later.
        SET(MKLML_PLAT "linux_shared")
        SET(MKLML_VER "Glibc225_vsErf_mklml_lnx_${TIME_VERSION}" CACHE STRING "" FORCE)
        SET(MKLML_URL "http://paddlepaddledeps.bj.bcebos.com/${MKLML_VER}.tgz" CACHE STRING "" FORCE)
        SET(MKLML_MD5 "MD5=5853727c2b941b0f0477693aaaa85405")
        SET(MKLML_LIB                 ${MKLML_LIB_DIR}/libmklml_intel.so)
        SET(MKLML_IOMP_LIB            ${MKLML_LIB_DIR}/libiomp5.so)
        SET(MKLML_SHARED_LIB          ${MKLML_LIB_DIR}/libmklml_intel.so)
        SET(MKLML_SHARED_IOMP_LIB     ${MKLML_LIB_DIR}/libiomp5.so)
    ENDIF()
ENDIF()

SET(MKLML_PROJECT       "extern_mklml")
MESSAGE(STATUS "MKLML_VER: ${MKLML_VER}, MKLML_URL: ${MKLML_URL}")
SET(MKLML_SOURCE_DIR    "${THIRD_PARTY_PATH}/mklml")
SET(MKLML_DOWNLOAD_DIR  "${CMAKE_SOURCE_DIR}/third-party/mklml")

ExternalProject_Add(
    ${MKLML_PROJECT}
    ${EXTERNAL_PROJECT_LOG_ARGS}
    PREFIX                 ${MKLML_SOURCE_DIR}
    URL                    ${MKLML_URL}
    URL_HASH               ${MKLML_MD5}
    DOWNLOAD_DIR          ${MKLML_DOWNLOAD_DIR}
    DOWNLOAD_NO_PROGRESS  1
    CONFIGURE_COMMAND     ""
    BUILD_COMMAND         ""
    UPDATE_COMMAND ""
    INSTALL_COMMAND
        ${CMAKE_COMMAND} -E copy_directory ${MKLML_SOURCE_DIR}/src/extern_mklml/include ${MKLML_INC_DIR} &&
        ${CMAKE_COMMAND} -E copy_directory ${MKLML_SOURCE_DIR}/src/extern_mklml/lib ${MKLML_LIB_DIR}
)

IF(NOT WIN32 AND NOT LITE_WITH_SW)
    add_compile_options(-m64)
ENDIF()
INCLUDE_DIRECTORIES(${MKLML_INC_DIR})

IF(WITH_STATIC_MKL)
    ADD_LIBRARY(mklml_intel_lp64 STATIC IMPORTED GLOBAL)
    SET_PROPERTY(TARGET mklml_intel_lp64 PROPERTY IMPORTED_LOCATION ${MKLML_LP64_LIB})
    ADD_DEPENDENCIES(mklml_intel_lp64 ${MKLML_PROJECT})

    ADD_LIBRARY(mklml_intel_thread STATIC IMPORTED GLOBAL)
    SET_PROPERTY(TARGET mklml_intel_thread PROPERTY IMPORTED_LOCATION ${MKLML_THREAD_LIB})
    ADD_DEPENDENCIES(mklml_intel_thread ${MKLML_PROJECT})

    ADD_LIBRARY(mklml_core STATIC IMPORTED GLOBAL)
    SET_PROPERTY(TARGET mklml_core PROPERTY IMPORTED_LOCATION ${MKLML_CORE_LIB})
    ADD_DEPENDENCIES(mklml_core ${MKLML_PROJECT})

    SET(MKLML_LIBRARIES mklml_intel_lp64 mklml_intel_thread mklml_core CACHE INTERNAL "Intel(R) MKLML Libraries")

    IF(WIN32)
        ADD_LIBRARY(mklml_iomp5 SHARED IMPORTED GLOBAL)
        SET_PROPERTY(TARGET mklml_iomp5 PROPERTY IMPORTED_LOCATION ${MKLML_SHARED_IOMP_LIB})
        SET_PROPERTY(TARGET mklml_iomp5 PROPERTY IMPORTED_IMPLIB ${MKLML_IOMP_LIB})
        ADD_DEPENDENCIES(mklml_iomp5 ${MKLML_PROJECT})
        set(MKLML_LIBRARIES "${MKLML_LIBRARIES};mklml_iomp5" CACHE INTERNAL "Intel(R) MKLML Libraries")
    ENDIF()
ELSE(WITH_STATIC_MKL)
    ADD_LIBRARY(mklml SHARED IMPORTED GLOBAL)
    SET_PROPERTY(TARGET mklml PROPERTY IMPORTED_LOCATION ${MKLML_LIB})
    ADD_DEPENDENCIES(mklml ${MKLML_PROJECT})

    SET(MKLML_LIBRARIES mklml CACHE INTERNAL "Intel(R) MKLML Libraries")
ENDIF(WITH_STATIC_MKL)