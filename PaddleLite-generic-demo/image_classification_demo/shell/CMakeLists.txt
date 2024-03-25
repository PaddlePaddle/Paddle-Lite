cmake_minimum_required(VERSION 3.10)

if(TARGET_OS STREQUAL "linux")
  set(CMAKE_SYSTEM_NAME Linux)
  if(TARGET_ABI STREQUAL "arm64")
    set(CMAKE_SYSTEM_PROCESSOR aarch64)
    set(CMAKE_C_COMPILER "aarch64-linux-gnu-gcc")
    set(CMAKE_CXX_COMPILER "aarch64-linux-gnu-g++")
  elseif(TARGET_ABI STREQUAL "armhf")
    set(CMAKE_SYSTEM_PROCESSOR arm)
    set(CMAKE_C_COMPILER "arm-linux-gnueabihf-gcc")
    set(CMAKE_CXX_COMPILER "arm-linux-gnueabihf-g++")
  elseif(TARGET_ABI STREQUAL "amd64")
    set(CMAKE_SYSTEM_PROCESSOR x86_64)
    set(CMAKE_C_COMPILER "gcc")
    set(CMAKE_CXX_COMPILER "g++")
  elseif(TARGET_ABI STREQUAL "loongarch64")
    set(CMAKE_SYSTEM_PROCESSOR loongarch64)
    set(CMAKE_C_COMPILER "gcc")
    set(CMAKE_CXX_COMPILER "g++")
  else()
    message(FATAL_ERROR "Unknown ABI ${TARGET_ABI}, only supports arm64, armhf and amd64 for Linux.")
    return()
  endif()
elseif(TARGET_OS STREQUAL "android")
  if(TARGET_ABI STREQUAL "arm64-v8a")
  elseif(TARGET_ABI STREQUAL "armeabi-v7a")
  else()
    message(FATAL_ERROR "Unknown ABI ${TARGET_ABI}, only supports arm64-v8a and armeabi-v7a for Android.")
    return()
  endif()
elseif(TARGET_OS STREQUAL "qnx")
  cmake_minimum_required(VERSION 3.22)
  set(CMAKE_SYSTEM_NAME QNX)
  set(QNX_HOST "$ENV{QNX_HOST}")
  set(QNX_TARGET "$ENV{QNX_TARGET}")
  if(NOT QNX_HOST)
    message(FATAL_ERROR "Environment variable QNX_HOST not set!")
  endif()
  if(NOT QNX_TARGET)
    message(FATAL_ERROR "Environment variable QNX_TARGET not set!")
  endif()
  if(TARGET_ABI STREQUAL "arm64")
    set(QNX_COMPILER_TARGET gcc_ntoaarch64le_cxx)
    set(QNX_PROCESSOR aarch64)
  elseif(TARGET_ABI STREQUAL "arm")
    set(QNX_COMPILER_TARGET gcc_ntoarmv7le_cxx)
    set(QNX_PROCESSOR armle-v7)
  else()
    message(FATAL_ERROR "Unknown ABI ${TARGET_ABI}, only supports arm64 and arm for QNX.")
    return()
  endif()
  set(CMAKE_C_COMPILER ${QNX_HOST}/usr/bin/qcc)
  set(CMAKE_C_COMPILER_TARGET ${QNX_COMPILER_TARGET})
  set(CMAKE_CXX_COMPILER ${QNX_HOST}/usr/bin/q++)
  set(CMAKE_CXX_COMPILER_TARGET ${QNX_COMPILER_TARGET})
  set(CMAKE_ASM_COMPILER ${QNX_HOST}/usr/bin/qcc -V${QNX_COMPILER_TARGET})
  set(CMAKE_ASM_DEFINE_FLAG "-Wa,--defsym,")
else()
  message(FATAL_ERROR "Unknown OS ${TARGET_OS}.")
  return()
endif()

project(demo)
message(STATUS "TARGET OS: ${TARGET_OS}")
message(STATUS "TARGET ABI: ${TARGET_ABI}")
message(STATUS "PADDLE LITE DIR: ${PADDLE_LITE_DIR}")
message(STATUS "OpenCV DIR: ${OpenCV_DIR}")

include_directories(${PADDLE_LITE_DIR}/${TARGET_OS}/${TARGET_ABI}/include)
link_directories(${PADDLE_LITE_DIR}/${TARGET_OS}/${TARGET_ABI}/lib)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wl,--allow-shlib-undefined")
if(TARGET_OS STREQUAL "qnx")
  add_definitions(-D_QNX_SOURCE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -V${QNX_COMPILER_TARGET} -fPIC -D_QNX_SOURCE=1")
else()
  if(TARGET_OS STREQUAL "linux")
    if(TARGET_ABI STREQUAL "arm64")
      set(CMAKE_CXX_FLAGS "-march=armv8-a ${CMAKE_CXX_FLAGS}")
      set(CMAKE_C_FLAGS "-march=armv8-a ${CMAKE_C_FLAGS}")
    elseif(TARGET_ABI STREQUAL "armhf")
      set(CMAKE_CXX_FLAGS "-march=armv7-a -mfloat-abi=hard -mfpu=neon-vfpv4 ${CMAKE_CXX_FLAGS}")
      set(CMAKE_C_FLAGS "-march=armv7-a -mfloat-abi=hard -mfpu=neon-vfpv4 ${CMAKE_C_FLAGS}" )
    endif()
  endif()
  find_package(OpenMP REQUIRED)
  if(OpenMP_FOUND OR OpenMP_CXX_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    message(STATUS "Found OpenMP ${OpenMP_VERSION} ${OpenMP_CXX_VERSION}")
    message(STATUS "OpenMP C flags:  ${OpenMP_C_FLAGS}")
    message(STATUS "OpenMP CXX flags:  ${OpenMP_CXX_FLAGS}")
    message(STATUS "OpenMP OpenMP_CXX_LIB_NAMES:  ${OpenMP_CXX_LIB_NAMES}")
    message(STATUS "OpenMP OpenMP_CXX_LIBRARIES:  ${OpenMP_CXX_LIBRARIES}")
  else()
    message(FATAL_ERROR "OpenMP not found!")
    return()
  endif()
endif()

if(TARGET_OS STREQUAL "android")
  set(OpenCV_DIR "${OpenCV_DIR}/${TARGET_OS}/${TARGET_ABI}/sdk/native/jni")
else()
  set(OpenCV_DIR "${OpenCV_DIR}/${TARGET_OS}/${TARGET_ABI}/lib/cmake/opencv4")
endif()
find_package(OpenCV REQUIRED)
if(OpenCV_FOUND OR OpenCV_CXX_FOUND)
    include_directories(${OpenCV_INCLUDE_DIRS})
    message(STATUS "OpenCV library status:")
    message(STATUS "    version: ${OpenCV_VERSION}")
    message(STATUS "    libraries: ${OpenCV_LIBS}")
    message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
else()
    message(FATAL_ERROR "OpenCV not found!")
    return()
endif()

add_executable(demo demo.cc)

if(NOT DEFINED USE_FULL_API)
  set(USE_FULL_API TRUE)
endif()

add_library(
  # Sets the name of the library.
  paddle_api_shared
  # Sets the library as a shared library.
  SHARED
  # Provides a relative path to your source file(s).
  IMPORTED)

if(USE_FULL_API)
  add_definitions(-DUSE_FULL_API)
  set_target_properties(
    # Specifies the target library.
    paddle_api_shared
    # Specifies the parameter you want to define.
    PROPERTIES
      IMPORTED_LOCATION
      ${PADDLE_LITE_DIR}/${TARGET_OS}/${TARGET_ABI}/lib/libpaddle_full_api_shared.so
      # Provides the path to the library you want to import.
  )
else()
  set_target_properties(
    # Specifies the target library.
    paddle_api_shared
    # Specifies the parameter you want to define.
    PROPERTIES
      IMPORTED_LOCATION
      ${PADDLE_LITE_DIR}/${TARGET_OS}/${TARGET_ABI}/lib/libpaddle_light_api_shared.so
      # Provides the path to the library you want to import.
  )
endif()

target_link_libraries(demo paddle_api_shared ${OpenCV_LIBS})
