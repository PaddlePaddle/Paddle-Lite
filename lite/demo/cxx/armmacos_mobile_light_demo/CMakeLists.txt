cmake_minimum_required(VERSION 2.8)
project(macos_mobilenet_v1)
set(TARGET mobilenet_light_api)

# 0. find Apple lib for Metal
if (APPLE AND METAL)
  message(STATUS "set METAL=ON")
  add_definitions("-DMETAL")
  find_library(METAL_LIBRARY Metal REQUIRED)
  find_library(GRAPHIC CoreGraphics REQUIRED)
  find_library(MPS_LIBRARY MetalPerformanceShaders REQUIRED)
endif()

# 1. link Paddle-Lite directory
link_directories("${PROJECT_SOURCE_DIR}/../../../cxx/lib")
include_directories("${PROJECT_SOURCE_DIR}/../../../cxx/include")

# 2. compile options 
add_definitions(-std=c++11 -O3 -pthread)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR})

# 3. add executable output
add_executable(${TARGET} mobilenetv1_light_api.cc)

if (APPLE AND METAL)
  target_link_libraries(${TARGET} ${METAL_LIBRARY} ${GRAPHIC} ${MPS_LIBRARY})
endif()
target_link_libraries(${TARGET} paddle_light_api_shared)
target_link_libraries(${TARGET} -ldl)

###############################################################
# How to use one of static libaray:                           #
#  `libpaddle_api_light_bundled.a`                            #
###############################################################
# Note: default use lite's shared library.                    #
###############################################################
# 1. Comment above line using `libpaddle_light_api_shared.dylib`
# 2. Undo comment below line using `libpaddle_api_light_bundled.a`

#target_link_libraries(${TARGET} ${LITE_DIR}/lib/libpaddle_api_light_bundled.a)
