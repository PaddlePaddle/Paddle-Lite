FILE(GLOB CV_SRC ${CMAKE_CURRENT_SOURCE_DIE}/cv/*.cc)
FILE(GLOB FPGA_CV_SRC ${CMAKE_CURRENT_SOURCE_DIE}/cv/fpga/*.cc)
set(CUSTOM_SRC logging.cc replace_stl/stream.cc)
set(BASIC_UTILS_SRC any.cc string.cc)

add_library(utils OBJECT ${BASIC_UTILS_SRC} ${CUSTOM_SRC})
