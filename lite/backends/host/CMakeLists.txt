FILE(GLOB HOST_MATH_SRC ${CMAKE_CURRENT_SOURCE_DIR}/math/*.cc)
add_library(host_math OBJECT ${HOST_MATH_SRC} target_wrapper.cc)
