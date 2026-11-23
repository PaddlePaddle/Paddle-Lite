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
# sve2 arm math source code in sve/ directory
FILE(GLOB SVE_ARM_MATH_SRC ${CMAKE_CURRENT_SOURCE_DIR}/sve/*.cc)

if(LITE_WITH_ARM82_FP16)
  set(ARM_MATH_SRC ${ARM_MATH_SRC} ${FP16_ARM_MATH_SRC})
endif()

# Default math_arm TUs keep CMAKE_*_FLAGS (-march=armv8.2-a+fp16+nolse). Do not
# append SVE sources into that library: +sve2 is per-TU so NEON objects stay
# free of SVE opcodes (Pixel 6 SIGILL if they do not).
set(_SVE2_MARCH "-march=armv8.2-a+sve2+fp16+dotprod+i8mm+nolse")
if(LITE_WITH_ARM8_SVE2)
  lite_cc_library(math_arm_sve SRCS ${SVE_ARM_MATH_SRC})
  set_property(TARGET math_arm_sve PROPERTY POSITION_INDEPENDENT_CODE ON)
  target_compile_options(math_arm_sve PRIVATE ${_SVE2_MARCH})
  set_source_files_properties(${SVE_ARM_MATH_SRC} PROPERTIES COMPILE_FLAGS ${_SVE2_MARCH})
  message(STATUS "math_arm_sve COMPILE_OPTIONS: ${_SVE2_MARCH}")
  lite_cc_library(math_arm SRCS ${ARM_MATH_SRC} DEPS math_arm_sve)
else()
  lite_cc_library(math_arm SRCS ${ARM_MATH_SRC})
endif()
