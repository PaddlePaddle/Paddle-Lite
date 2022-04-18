# source code for unit tests
FILE(GLOB_RECURSE UNIT_TEST_SRC ${CMAKE_CURRENT_SOURCE_DIR}/*test*)

# self-defind log source code
FILE(GLOB LOG_SRC ${CMAKE_CURRENT_SOURCE_DIR}/log/*.cc)
LIST(REMOVE_ITEM LOG_SRC ${UNIT_TEST_SRC})

# cv library source code
FILE(GLOB CV_ARM_SRC ${CMAKE_CURRENT_SOURCE_DIR}/cv/*.cc)
FILE(GLOB CV_FPGA_SRC ${CMAKE_CURRENT_SOURCE_DIR}/cv/fpga/*.cc)
LIST(REMOVE_ITEM CV_ARM_SRC ${UNIT_TEST_SRC})
LIST(REMOVE_ITEM CV_FPGA_SRC ${UNIT_TEST_SRC})

# self-defined stl source code
FILE(GLOB STL_SRC ${CMAKE_CURRENT_SOURCE_DIR}/replace_stl/*.cc)
LIST(REMOVE_ITEM STL_SRC ${UNIT_TEST_SRC})

# source code in current directory
FILE(GLOB BASE_UTILS_SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.cc)
LIST(REMOVE_ITEM BASE_UTILS_SRC ${UNIT_TEST_SRC})


set(UTILS_SRC ${BASE_UTILS_SRC})
set(UTILS_DEPS "")

# 1. self-defined stl will be included in tiny_publish mode
if(LITE_ON_TINY_PUBLISH OR LITE_ON_MODEL_OPTIMIZE_TOOL)
  set(UTILS_SRC ${UTILS_SRC} ${STL_SRC})
endif()

# 2.opencv-source code will be included if LITE_WITH_CV
if(LITE_WITH_CV AND LITE_WITH_ARM)
  set(UTILS_SRC ${UTILS_SRC} ${CV_ARM_SRC})
  if(LITE_WITH_FPGA)
    set(UTILS_SRC ${UTILS_SRC} ${CV_FPGA_SRC})
    set(UTILS_DEPS ${UTILS_DEPS} ${kernel_fpga})
  endif()
endif()

# 3. self-defined log will be included in tiny_publish mode
if(LITE_WITH_ARM OR LITE_ON_MODEL_OPTIMIZE_TOOL OR LITE_WITH_PYTHON OR LITE_WITH_XPU OR (NOT LITE_WITH_LOG))
  set(UTILS_SRC ${UTILS_SRC} ${LOG_SRC})
  lite_cc_test(test_logging SRCS ${CMAKE_CURRENT_SOURCE_DIR}/log/logging_test.cc)
else()
   set(UTILS_DEPS ${UTILS_DEPS} glog)
endif()


lite_cc_library(utils SRCS ${UTILS_SRC} DEPS ${UTILS_DEPS})



###########################################################
#  unit tests
###########################################################
lite_cc_test(test_varient SRCS varient_test.cc)
lite_cc_test(test_utils_string SRCS string_test.cc)
lite_cc_test(test_fast_type_id SRCS fast_type_id_test.cc)
# fp16 unit test
if (WITH_TESTING)
    if (LITE_WITH_CUDA)
      nv_test(float16_gpu_test SRCS float16_test.cu)
    endif()
    lite_cc_test(float16_test SRCS float16_test.cc)
endif()

# timer unit test
lite_cc_test(test_timer SRCS timer_test.cc)
