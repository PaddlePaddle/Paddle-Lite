# This file is part of the ios-cmake project. It was retrieved from
# https://github.com/cristeab/ios-cmake.git, which is a fork of
# https://code.google.com/p/ios-cmake/. Which in turn is based off of
# the Platform/Darwin.cmake and Platform/UnixPaths.cmake files which
# are included with CMake 2.8.4
#
# The ios-cmake project is licensed under the new BSD license.
#
# Copyright (c) 2014, Bogdan Cristea and LTE Engineering Software,
# Kitware, Inc., Insight Software Consortium.  All rights reserved.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
# 1. Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# This file is based off of the Platform/Darwin.cmake and
# Platform/UnixPaths.cmake files which are included with CMake 2.8.4
# It has been altered for iOS development.
#
# Updated by Alex Stewart (alexs.mac@gmail.com)
#
# *****************************************************************************
#      Now maintained by Alexander Widerberg (widerbergaren [at] gmail.com)
#                      under the BSD-3-Clause license
#                   https://github.com/leetal/ios-cmake
# *****************************************************************************

## Lite settings
set(PLATFORM "MACOS")

if(ARM_TARGET_ARCH_ABI STREQUAL "armv8"
    OR ARM_TARGET_ARCH_ABI STREQUAL "arm64-v8a")
  set(ARCHS "arm64")
endif()

if(PLATFORM STREQUAL "MACOS" AND ARCHS STREQUAL "armv7")
  message(FATAL_ERROR "Can not build arm macos with armv7")
endif()


# TODO(xxx): enable omp on ios
set(LITE_WITH_OPENMP OFF CACHE STRING "Disable OpenMP when cross-compiling for Android and iOS" FORCE)
set(ARM_TARGET_LANG "clang" CACHE STRING "Force use clang on IOS" FORCE)

add_definitions(-DLITE_WITH_IPHONE)
## End lite settings

# Fix for PThread library not in path
set(CMAKE_THREAD_LIBS_INIT "-lpthread")
set(CMAKE_HAVE_THREADS_LIBRARY 1)
set(CMAKE_USE_WIN32_THREADS_INIT 0)
set(CMAKE_USE_PTHREADS_INIT 1)

# Cache what generator is used
set(USED_CMAKE_GENERATOR "${CMAKE_GENERATOR}" CACHE STRING "Expose CMAKE_GENERATOR" FORCE)

if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.14")
  set(MODERN_CMAKE YES)
  message(STATUS "Merging integrated CMake 3.14+ iOS,tvOS,watchOS,macOS toolchain(s) with this toolchain!")
endif()

# Get the Xcode version being used.
execute_process(COMMAND xcodebuild -version
  OUTPUT_VARIABLE XCODE_VERSION
  ERROR_QUIET
  OUTPUT_STRIP_TRAILING_WHITESPACE)
string(REGEX MATCH "Xcode [0-9\\.]+" XCODE_VERSION "${XCODE_VERSION}")
string(REGEX REPLACE "Xcode ([0-9\\.]+)" "\\1" XCODE_VERSION "${XCODE_VERSION}")
message(STATUS "Building with Xcode version: ${XCODE_VERSION}")

# Unset the FORCE on cache variables if in try_compile()
set(FORCE_CACHE FORCE)
get_property(_CMAKE_IN_TRY_COMPILE GLOBAL PROPERTY IN_TRY_COMPILE)
if(_CMAKE_IN_TRY_COMPILE)
  unset(FORCE_CACHE)
endif()
set(PLATFORM_INT "${PLATFORM}" CACHE STRING "Type of platform for which the build targets.")

# Determine the platform name and architectures for use in xcodebuild commands
# from the specified PLATFORM name.
if(PLATFORM_INT STREQUAL "MACOS")
  set(SDK_NAME macosx)
  if(NOT ARCHS)
    set(ARCHS armv8)
  endif()
message(STATUS "Configuring ${SDK_NAME} build for platform: ${PLATFORM_INT}, architecture(s): ${ARCHS}")

# If user did not specify the SDK root to use, then query xcodebuild for it.
execute_process(COMMAND xcodebuild -version -sdk ${SDK_NAME} Path
    OUTPUT_VARIABLE CMAKE_OSX_SYSROOT_INT
    ERROR_QUIET
    OUTPUT_STRIP_TRAILING_WHITESPACE)
if (NOT DEFINED CMAKE_OSX_SYSROOT_INT AND NOT DEFINED CMAKE_OSX_SYSROOT)
  message(SEND_ERROR "Please make sure that Xcode is installed and that the toolchain"
  "is pointing to the correct path. Please run:"
  "sudo xcode-select -s /Applications/Xcode.app/Contents/Developer"
  "and see if that fixes the problem for you.")
  message(FATAL_ERROR "Invalid CMAKE_OSX_SYSROOT: ${CMAKE_OSX_SYSROOT} "
  "does not exist.")
elseif(DEFINED CMAKE_OSX_SYSROOT)
  message(STATUS "Using SDK: ${CMAKE_OSX_SYSROOT} for platform: ${PLATFORM_INT} when checking compatibility")
elseif(DEFINED CMAKE_OSX_SYSROOT_INT)
   message(STATUS "Using SDK: ${CMAKE_OSX_SYSROOT_INT} for platform: ${PLATFORM_INT}")
   set(CMAKE_OSX_SYSROOT "${CMAKE_OSX_SYSROOT_INT}" CACHE INTERNAL "")
endif()

# Set Xcode property for SDKROOT as well if Xcode generator is used
if(USED_CMAKE_GENERATOR MATCHES "Xcode")
  set(CMAKE_OSX_SYSROOT "${SDK_NAME}" CACHE INTERNAL "")
  if(NOT DEFINED CMAKE_XCODE_ATTRIBUTE_DEVELOPMENT_TEAM)
    set(CMAKE_XCODE_ATTRIBUTE_DEVELOPMENT_TEAM 123456789A CACHE INTERNAL "")
  endif()
endif()

# Specify minimum version of deployment target.
if(NOT DEFINED DEPLOYMENT_TARGET)
  set(DEPLOYMENT_TARGET "10.0"
            CACHE STRING "Minimum SDK version to build for." )
  message(STATUS "Using the default min-version since DEPLOYMENT_TARGET not provided!")
endif()

# Use bitcode or not
if(NOT DEFINED ENABLE_BITCODE)
  message(STATUS "Disabling bitcode support by default on simulators. ENABLE_BITCODE not provided for override!")
  set(ENABLE_BITCODE FALSE)
endif()
set(ENABLE_BITCODE_INT ${ENABLE_BITCODE} CACHE BOOL "Whether or not to enable bitcode" ${FORCE_CACHE})
# Use ARC or not
if(NOT DEFINED ENABLE_ARC)
  # Unless specified, enable ARC support by default
  set(ENABLE_ARC TRUE)
  message(STATUS "Enabling ARC support by default. ENABLE_ARC not provided!")
endif()
set(ENABLE_ARC_INT ${ENABLE_ARC} CACHE BOOL "Whether or not to enable ARC" ${FORCE_CACHE})
# Use hidden visibility or not
if(NOT DEFINED ENABLE_VISIBILITY)
  # Unless specified, disable symbols visibility by default
  set(ENABLE_VISIBILITY FALSE)
  message(STATUS "Hiding symbols visibility by default. ENABLE_VISIBILITY not provided!")
endif()
set(ENABLE_VISIBILITY_INT ${ENABLE_VISIBILITY} CACHE BOOL "Whether or not to hide symbols (-fvisibility=hidden)" ${FORCE_CACHE})
# Get the SDK version information.
execute_process(COMMAND xcodebuild -sdk ${CMAKE_OSX_SYSROOT} -version SDKVersion
  OUTPUT_VARIABLE SDK_VERSION
  ERROR_QUIET
  OUTPUT_STRIP_TRAILING_WHITESPACE)

# Find the Developer root for the specific iOS platform being compiled for
# from CMAKE_OSX_SYSROOT.  Should be ../../ from SDK specified in
# CMAKE_OSX_SYSROOT. There does not appear to be a direct way to obtain
# this information from xcrun or xcodebuild.
if (NOT DEFINED CMAKE_DEVELOPER_ROOT AND NOT USED_CMAKE_GENERATOR MATCHES "Xcode")
  get_filename_component(PLATFORM_SDK_DIR ${CMAKE_OSX_SYSROOT} PATH)
  get_filename_component(CMAKE_DEVELOPER_ROOT ${PLATFORM_SDK_DIR} PATH)

  if (NOT DEFINED CMAKE_DEVELOPER_ROOT)
    message(FATAL_ERROR "Invalid CMAKE_DEVELOPER_ROOT: "
      "${CMAKE_DEVELOPER_ROOT} does not exist.")
  endif()
endif()
# Find the C & C++ compilers for the specified SDK.
if(NOT CMAKE_C_COMPILER)
  execute_process(COMMAND xcrun -sdk ${CMAKE_OSX_SYSROOT} -find clang
    OUTPUT_VARIABLE CMAKE_C_COMPILER
    ERROR_QUIET
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  message(STATUS "Using C compiler: ${CMAKE_C_COMPILER}")
endif()
if(NOT CMAKE_CXX_COMPILER)
  execute_process(COMMAND xcrun -sdk ${CMAKE_OSX_SYSROOT} -find clang++
    OUTPUT_VARIABLE CMAKE_CXX_COMPILER
    ERROR_QUIET
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  message(STATUS "Using CXX compiler: ${CMAKE_CXX_COMPILER}")
endif()
# Find (Apple's) libtool.
execute_process(COMMAND xcrun -sdk ${CMAKE_OSX_SYSROOT} -find libtool
  OUTPUT_VARIABLE BUILD_LIBTOOL
  ERROR_QUIET
  OUTPUT_STRIP_TRAILING_WHITESPACE)
message(STATUS "Using libtool: ${BUILD_LIBTOOL}")
# Configure libtool to be used instead of ar + ranlib to build static libraries.
# This is required on Xcode 7+, but should also work on previous versions of
# Xcode.
set(CMAKE_C_CREATE_STATIC_LIBRARY
  "${BUILD_LIBTOOL} -static -o <TARGET> <LINK_FLAGS> <OBJECTS> ")
set(CMAKE_CXX_CREATE_STATIC_LIBRARY
  "${BUILD_LIBTOOL} -static -o <TARGET> <LINK_FLAGS> <OBJECTS> ")
# Get the version of Darwin (OS X) of the host.
execute_process(COMMAND uname -r
  OUTPUT_VARIABLE CMAKE_HOST_SYSTEM_VERSION
  ERROR_QUIET
  OUTPUT_STRIP_TRAILING_WHITESPACE)

  # Legacy code path prior to CMake 3.14
  set(CMAKE_SYSTEM_NAME Darwin CACHE INTERNAL "" ${FORCE_CACHE})
endif()
# Standard settings.
set(CMAKE_SYSTEM_VERSION ${SDK_VERSION} CACHE INTERNAL "")
set(UNIX TRUE CACHE BOOL "")
set(APPLE TRUE CACHE BOOL "")
set(ARMMACOS TRUE CACHE BOOL "")
set(CMAKE_AR ar CACHE FILEPATH "" FORCE)
set(CMAKE_RANLIB ranlib CACHE FILEPATH "" FORCE)
set(CMAKE_STRIP strip CACHE FILEPATH "" FORCE)
# Set the architectures for which to build.
set(CMAKE_OSX_ARCHITECTURES ${ARCHS} CACHE STRING "Build architecture for iOS")
# Change the type of target generated for try_compile() so it'll work when cross-compiling
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
# All iOS/Darwin specific settings - some may be redundant.
set(CMAKE_SHARED_LIBRARY_PREFIX "lib")
set(CMAKE_SHARED_LIBRARY_SUFFIX ".dylib")
set(CMAKE_SHARED_MODULE_PREFIX "lib")
set(CMAKE_SHARED_MODULE_SUFFIX ".so")
set(CMAKE_C_COMPILER_ABI ELF)
set(CMAKE_CXX_COMPILER_ABI ELF)
set(CMAKE_C_HAS_ISYSROOT 1)
set(CMAKE_CXX_HAS_ISYSROOT 1)
set(CMAKE_MODULE_EXISTS 1)
set(CMAKE_DL_LIBS "")
set(CMAKE_C_OSX_COMPATIBILITY_VERSION_FLAG "-compatibility_version ")
set(CMAKE_C_OSX_CURRENT_VERSION_FLAG "-current_version ")
set(CMAKE_CXX_OSX_COMPATIBILITY_VERSION_FLAG "${CMAKE_C_OSX_COMPATIBILITY_VERSION_FLAG}")
set(CMAKE_CXX_OSX_CURRENT_VERSION_FLAG "${CMAKE_C_OSX_CURRENT_VERSION_FLAG}")

if(ARCHS MATCHES "((^|, )(arm64|arm64e|x86_64))+")
  set(CMAKE_C_SIZEOF_DATA_PTR 8)
  set(CMAKE_CXX_SIZEOF_DATA_PTR 8)
  if(ARCHS MATCHES "((^|, )(arm64|arm64e))+")
    set(CMAKE_SYSTEM_PROCESSOR "arm64")
  else()
    set(CMAKE_SYSTEM_PROCESSOR "x86_64")
  endif()
  message(STATUS "Using a data_ptr size of 8")
else()
  set(CMAKE_C_SIZEOF_DATA_PTR 4)
  set(CMAKE_CXX_SIZEOF_DATA_PTR 4)
  set(CMAKE_SYSTEM_PROCESSOR "arm")
  message(STATUS "Using a data_ptr size of 4")
endif()

message(STATUS "Building for minimum ${SDK_NAME} version: ${DEPLOYMENT_TARGET}"
               " (SDK version: ${SDK_VERSION})")

set(SDK_NAME_VERSION_FLAGS
  "-m${SDK_NAME}-version-min=${DEPLOYMENT_TARGET}")
set(CMAKE_CXX_FLAGS "-target arm64-apple-macos11 ${CMAKE_CXX_FLAGS}")
set(CMAKE_C_FLAGS "-target arm64-apple-macos11 ${CMAKE_C_FLAGS}")

message(STATUS "Version flags set to: ${SDK_NAME_VERSION_FLAGS}")
set(CMAKE_OSX_DEPLOYMENT_TARGET ${DEPLOYMENT_TARGET} CACHE STRING
    "Set CMake deployment target" ${FORCE_CACHE})

if(ENABLE_BITCODE_INT)
  set(BITCODE "-fembed-bitcode")
  set(CMAKE_XCODE_ATTRIBUTE_BITCODE_GENERATION_MODE bitcode CACHE INTERNAL "")
  message(STATUS "Enabling bitcode support.")
else()
  set(BITCODE "")
  set(CMAKE_XCODE_ATTRIBUTE_ENABLE_BITCODE NO CACHE INTERNAL "")
  message(STATUS "Disabling bitcode support.")
endif()

if(ENABLE_ARC_INT)
  set(FOBJC_ARC "-fobjc-arc")
  set(CMAKE_XCODE_ATTRIBUTE_CLANG_ENABLE_OBJC_ARC YES CACHE INTERNAL "")
  message(STATUS "Enabling ARC support.")
else()
  set(FOBJC_ARC "-fno-objc-arc")
  set(CMAKE_XCODE_ATTRIBUTE_CLANG_ENABLE_OBJC_ARC NO CACHE INTERNAL "")
  message(STATUS "Disabling ARC support.")
endif()

if(NOT ENABLE_VISIBILITY_INT)
  set(VISIBILITY "-fvisibility=hidden")
  set(CMAKE_XCODE_ATTRIBUTE_GCC_SYMBOLS_PRIVATE_EXTERN YES CACHE INTERNAL "")
  message(STATUS "Hiding symbols (-fvisibility=hidden).")
else()
  set(VISIBILITY "")
  set(CMAKE_XCODE_ATTRIBUTE_GCC_SYMBOLS_PRIVATE_EXTERN NO CACHE INTERNAL "")
endif()

#Check if Xcode generator is used, since that will handle these flags automagically
if(USED_CMAKE_GENERATOR MATCHES "Xcode")
  message(STATUS "Not setting any manual command-line buildflags, since Xcode is selected as generator.")
else()
  set(CMAKE_C_FLAGS
  "${SDK_NAME_VERSION_FLAGS} ${BITCODE} -fobjc-abi-version=2 ${FOBJC_ARC} ${CMAKE_C_FLAGS}")
  # Hidden visibilty is required for C++ on iOS.
  set(CMAKE_CXX_FLAGS
  "${SDK_NAME_VERSION_FLAGS} ${BITCODE} ${VISIBILITY} -fvisibility-inlines-hidden -fobjc-abi-version=2 ${FOBJC_ARC} ${CMAKE_CXX_FLAGS}")
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -O0 -g ${CMAKE_CXX_FLAGS_DEBUG}")
  set(CMAKE_CXX_FLAGS_MINSIZEREL "${CMAKE_CXX_FLAGS} -DNDEBUG -Os -ffast-math ${CMAKE_CXX_FLAGS_MINSIZEREL}")
  set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS} -DNDEBUG -O2 -g -ffast-math ${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -DNDEBUG -O3 -ffast-math ${CMAKE_CXX_FLAGS_RELEASE}")
  set(CMAKE_C_LINK_FLAGS "${SDK_NAME_VERSION_FLAGS} -Wl,-search_paths_first ${CMAKE_C_LINK_FLAGS}")
  set(CMAKE_CXX_LINK_FLAGS "${SDK_NAME_VERSION_FLAGS}  -Wl,-search_paths_first ${CMAKE_CXX_LINK_FLAGS}")

  # In order to ensure that the updated compiler flags are used in try_compile()
  # tests, we have to forcibly set them in the CMake cache, not merely set them
  # in the local scope.
  list(APPEND VARS_TO_FORCE_IN_CACHE
    CMAKE_C_FLAGS
    CMAKE_CXX_FLAGS
    CMAKE_CXX_FLAGS_DEBUG
    CMAKE_CXX_FLAGS_RELWITHDEBINFO
    CMAKE_CXX_FLAGS_MINSIZEREL
    CMAKE_CXX_FLAGS_RELEASE
    CMAKE_C_LINK_FLAGS
    CMAKE_CXX_LINK_FLAGS)
  foreach(VAR_TO_FORCE ${VARS_TO_FORCE_IN_CACHE})
    set(${VAR_TO_FORCE} "${${VAR_TO_FORCE}}" CACHE STRING "")
  endforeach()
endif()

set(CMAKE_PLATFORM_HAS_INSTALLNAME 1)
set(CMAKE_SHARED_LINKER_FLAGS "-rpath @executable_path/Frameworks -rpath @loader_path/Frameworks")
set(CMAKE_SHARED_LIBRARY_CREATE_C_FLAGS "-dynamiclib -Wl,-headerpad_max_install_names")
set(CMAKE_SHARED_MODULE_CREATE_C_FLAGS "-bundle -Wl,-headerpad_max_install_names")
set(CMAKE_SHARED_MODULE_LOADER_C_FLAG "-Wl,-bundle_loader,")
set(CMAKE_SHARED_MODULE_LOADER_CXX_FLAG "-Wl,-bundle_loader,")
set(CMAKE_FIND_LIBRARY_SUFFIXES ".tbd" ".dylib" ".so" ".a")
set(CMAKE_SHARED_LIBRARY_SONAME_C_FLAG "-install_name")

# Hack: if a new cmake (which uses CMAKE_INSTALL_NAME_TOOL) runs on an old
# build tree (where install_name_tool was hardcoded) and where
# CMAKE_INSTALL_NAME_TOOL isn't in the cache and still cmake didn't fail in
# CMakeFindBinUtils.cmake (because it isn't rerun) hardcode
# CMAKE_INSTALL_NAME_TOOL here to install_name_tool, so it behaves as it did
# before, Alex.
if(NOT DEFINED CMAKE_INSTALL_NAME_TOOL)
  find_program(CMAKE_INSTALL_NAME_TOOL install_name_tool)
endif(NOT DEFINED CMAKE_INSTALL_NAME_TOOL)

# Set the find root to the iOS developer roots and to user defined paths.
set(CMAKE_FIND_ROOT_PATH ${CMAKE_DEVELOPER_ROOT} ${CMAKE_OSX_SYSROOT_INT}
  ${CMAKE_PREFIX_PATH} CACHE STRING "Root path that will be prepended to all search paths")
# Default to searching for frameworks first.
set(CMAKE_FIND_FRAMEWORK FIRST)
# Set up the default search directories for frameworks.
set(CMAKE_FRAMEWORK_PATH
  ${CMAKE_DEVELOPER_ROOT}/Library/Frameworks
  ${CMAKE_DEVELOPER_ROOT}/Library/PrivateFrameworks
  ${CMAKE_OSX_SYSROOT_INT}/System/Library/Frameworks
  ${CMAKE_FRAMEWORK_PATH} CACHE STRING "Frameworks search paths")

# By default, search both the specified iOS SDK and the remainder of the host filesystem.
if(NOT CMAKE_FIND_ROOT_PATH_MODE_PROGRAM)
  set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM BOTH CACHE STRING "" ${FORCE_CACHE})
endif()
if(NOT CMAKE_FIND_ROOT_PATH_MODE_LIBRARY)
  set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY BOTH CACHE STRING "" ${FORCE_CACHE})
endif()
if(NOT CMAKE_FIND_ROOT_PATH_MODE_INCLUDE)
  set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE BOTH CACHE STRING "" ${FORCE_CACHE})
endif()
if(NOT CMAKE_FIND_ROOT_PATH_MODE_PACKAGE)
  set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE BOTH CACHE STRING "" ${FORCE_CACHE})
endif()

#
# Some helper-macros below to simplify and beautify the CMakeFile
#

# This little macro lets you set any XCode specific property.
macro(set_xcode_property TARGET XCODE_PROPERTY XCODE_VALUE XCODE_RELVERSION)
  set(XCODE_RELVERSION_I "${XCODE_RELVERSION}")
  if(XCODE_RELVERSION_I STREQUAL "All")
    set_property(TARGET ${TARGET} PROPERTY
    XCODE_ATTRIBUTE_${XCODE_PROPERTY} "${XCODE_VALUE}")
  else()
    set_property(TARGET ${TARGET} PROPERTY
    XCODE_ATTRIBUTE_${XCODE_PROPERTY}[variant=${XCODE_RELVERSION_I}] "${XCODE_VALUE}")
  endif()
endmacro(set_xcode_property)
# This macro lets you find executable programs on the host system.
macro(find_host_package)
  set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
  set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY NEVER)
  set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE NEVER)
  set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE NEVER)
  set(IOS FALSE)
  find_package(${ARGN})
  set(ARMMACOS TRUE)
  set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM BOTH)
  set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY BOTH)
  set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE BOTH)
  set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE BOTH)
endmacro(find_host_package)
