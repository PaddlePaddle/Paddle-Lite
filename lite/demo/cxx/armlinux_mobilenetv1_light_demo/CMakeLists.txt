cmake_minimum_required(VERSION 2.8)

set(TARGET mobilenetv1_light_api)

# Set ARMLINUX_ARCH_ABI to armv8 or armv7 or armv7hf
set(ARMLINUX_ARCH_ABI armv8)

# 1. path to Paddle-Lite lib
set(LITE_DIR "${PROJECT_SOURCE_DIR}/../../../cxx")

# 2. link Paddle-Lite directory
link_directories(${LITE_DIR}/lib)

include_directories(${LITE_DIR}/include)

# 3. compile options 
if(ARMLINUX_ARCH_ABI STREQUAL "armv8")
  set(CMAKE_C_COMPILER "aarch64-linux-gnu-gcc")
  set(CMAKE_CXX_COMPILER "aarch64-linux-gnu-g++")
elseif(ARMLINUX_ARCH_ABI STREQUAL "armv7")
  set(CMAKE_C_COMPILER "arm-linux-gnueabi-gcc")
  set(CMAKE_CXX_COMPILER "arm-linux-gnueabi-g++")
elseif(ARMLINUX_ARCH_ABI STREQUAL "armv7hf")
  set(CMAKE_C_COMPILER "arm-linux-gnueabihf-gcc")
  set(CMAKE_CXX_COMPILER "arm-linux-gnueabihf-g++")
else()
  message(FATAL_ERROR "Illegal ARMLINUX_ARCH_ABI: ${ARMLINUX_ARCH_ABI}")
endif()

message(STATUS "armlinux CMAKE_C_COMPILER: ${CMAKE_C_COMPILER}")
message(STATUS "armlinux CMAKE_CXX_COMPILER: ${CMAKE_CXX_COMPILER}")

add_definitions(-std=c++11 -O3 -pthread)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR})

# 4.add executable output
add_executable(${TARGET} ${TARGET}.cc)
target_link_libraries(${TARGET} -lpaddle_light_api_shared)
target_link_libraries(${TARGET} -ldl)

###############################################################
# How to use one of static libaray:                           #
#  `libpaddle_api_light_bundled.a`                            #
###############################################################
# Note: default use lite's shared library.                    #
###############################################################
# 1. Comment above line using `libpaddle_light_api_shared.so`
# 2. Undo comment below line using `libpaddle_api_light_bundled.a`

#target_link_libraries(${TARGET} ${LITE_DIR}/lib/libpaddle_api_light_bundled.a)
