if(NOT LITE_WITH_ARM)
  return()
endif()

#-----------------------------------------------------------------------------------------------#
# scipts to generate head files
set(script_dir ${CMAKE_CURRENT_SOURCE_DIR}/../../../tools/)
message(STATUS "generating arm dotprod code")
find_package(PythonInterp REQUIRED)
execute_process(COMMAND ${PYTHON_EXECUTABLE} ${script_dir}/convert_arm_sdot_to_machine_code.py
        "--input_file=${CMAKE_CURRENT_SOURCE_DIR}/dotprod/__gemm_sdot_meta__.h"
        "--output_file=${CMAKE_CURRENT_SOURCE_DIR}/dotprod/gemm_sdot.h"
        RESULT_VARIABLE gen_code_ret_v8)
if (NOT ${gen_code_ret_v8} STREQUAL "0")
  message(FATAL_ERROR "generating dotprod code quit with error: ${gen_code_ret_v8}")
endif ()
execute_process(COMMAND ${PYTHON_EXECUTABLE} ${script_dir}/convert_arm_vsdot_to_machine_code.py
        "--input_file=${CMAKE_CURRENT_SOURCE_DIR}/dotprod/__gemm_vsdot_meta__.h"
        "--output_file=${CMAKE_CURRENT_SOURCE_DIR}/dotprod/gemm_vsdot.h"
        RESULT_VARIABLE gen_code_ret_v7)
if (NOT ${gen_code_ret_v7} STREQUAL "0")
  message(FATAL_ERROR "generating dotprod code quit with error: ${gen_code_ret_v7}")
endif ()
#-----------------------------------------------------------------------------------------------#



# arm math source code in current directory
FILE(GLOB ARM_MATH_SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.cc)
# fp16 arm math source code in fp16/ directory
FILE(GLOB FP16_ARM_MATH_SRC ${CMAKE_CURRENT_SOURCE_DIR}/fp16/*.cc)
# sve2 arm math source code in sve2/ directory
FILE(GLOB SVE_ARM_MATH_SRC ${CMAKE_CURRENT_SOURCE_DIR}/sve/*.cc)

if(LITE_WITH_ARM82_FP16)
  set(ARM_MATH_SRC ${ARM_MATH_SRC} ${FP16_ARM_MATH_SRC})
endif()

if(LITE_WITH_ARM8_SVE2)
   set(ARM_MATH_SRC ${ARM_MATH_SRC} ${SVE_ARM_MATH_SRC})
endif()

lite_cc_library(math_arm SRCS ${ARM_MATH_SRC})
