cmake_minimum_required(VERSION 2.8)
set (CMAKE_CXX_STANDARD 11)

# Project's name

if(NOT DEFINED LITE_ROOT)
  message(FATAL_ERROR "please set LITE_ROOT with
                 -DLITE_ROOT=/path/to/your/build.lite.android.armv7.gcc/")
endif()

project(demo_trainer)
# Set the output folder where your program will be created
set(PADDLE_BINARY_DIR ${PADDLE_SOURCE_DIR}/bin)
set(EXECUTABLE_OUTPUT_PATH ${PADDLE_BINARY_DIR})
set(LIBRARY_OUTPUT_PATH ${PADDLE_BINARY_DIR})

# The following folder will be included
include_directories("include")
include_directories("${LITE_ROOT}/inference_lite_lib.android.armv7/cxx/include")

add_executable(demo_trainer ${PROJECT_SOURCE_DIR}/demo_trainer.cc ${PROJECT_SOURCE_DIR}/data_reader.cc)

TARGET_LINK_LIBRARIES(demo_trainer
"${LITE_ROOT}/inference_lite_lib.android.armv7/cxx/lib/libpaddle_full_api_shared.so")
