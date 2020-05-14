cmake_minimum_required(VERSION 2.8)

set(TARGET mobilenet_full_api)

# 1. path to Paddle-Lite lib and mklml lib
set(LITE_DIR "${PROJECT_SOURCE_DIR}/../../../cxx")
set(MKLML_DIR "${PROJECT_SOURCE_DIR}/../../../third_party/mklml/")

if (WIN32)
  add_definitions("/DGOOGLE_GLOG_DLL_DECL=")
  option(MSVC_STATIC_CRT "use static C Runtime library by default" ON)
  if (MSVC_STATIC_CRT)
    set(CMAKE_C_FLAGS_DEBUG   "${CMAKE_C_FLAGS_DEBUG} /bigobj /MTd")
    set(CMAKE_C_FLAGS_RELEASE  "${CMAKE_C_FLAGS_RELEASE} /bigobj /MT")
    set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG} /bigobj /MTd")
    set(CMAKE_CXX_FLAGS_RELEASE   "${CMAKE_CXX_FLAGS_RELEASE} /bigobj /MT")
  endif()
endif()

# 2. link mklml and Paddle-Lite directory
link_directories(${LITE_DIR}/lib ${MKLML_DIR}/lib)
include_directories(${LITE_DIR}/include/ ${MKLML_DIR}/include)

# 3. compile options 
if (NOT WIN32)
  add_definitions(-std=c++11 -g -O3 -pthread)
  set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR})
endif()

# 4.add executable output
add_executable(${TARGET} ${TARGET}.cc)
if (WIN32)
  set(MATH_LIB ${MKLML_DIR}/lib/mklml${CMAKE_STATIC_LIBRARY_SUFFIX}
      ${MKLML_DIR}/lib/libiomp5md${CMAKE_STATIC_LIBRARY_SUFFIX})


  target_link_libraries(${TARGET} libpaddle_api_full_bundled.lib)
  target_link_libraries(${TARGET} shlwapi.lib)
  target_link_libraries(${TARGET} ${MATH_LIB})

   add_custom_command(TARGET ${TARGET} POST_BUILD
       COMMAND ${CMAKE_COMMAND} -E copy ${MKLML_DIR}/lib/mklml.dll ${CMAKE_BINARY_DIR}/Release
       COMMAND ${CMAKE_COMMAND} -E copy ${MKLML_DIR}/lib/libiomp5md.dll ${CMAKE_BINARY_DIR}/Release
   )
else()
    target_link_libraries(${TARGET} -lpaddle_full_api_shared)
    target_link_libraries(${TARGET} -liomp5)
    target_link_libraries(${TARGET} -ldl)
endif()
