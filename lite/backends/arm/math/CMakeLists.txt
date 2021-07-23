#if(NOT (LITE_WITH_LIGHT_WEIGHT_FRAMEWORK AND LITE_WITH_ARM))
#  return()
#endif()

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

set(HAS_ARM_MATH_LIB_DIR OFF)
# will search name as "libmath_arm.${os}.${abi}.${lang}.a"
if(ARM_MATH_LIB_DIR AND EXISTS "${ARM_MATH_LIB_DIR}")
  set(arm_math_name "")
  if(ARM_TARGET_OS STREQUAL "android")
    if(ARM_TARGET_ARCH_ABI STREQUAL "armv8")
      set(arm_math_name "math_arm.android.armv8")
    elseif(ARM_TARGET_ARCH_ABI STREQUAL "armv7")
      set(arm_math_name "math_arm.android.armv7")
    endif()
  endif()

  if(ARM_TARGET_OS STREQUAL "armlinux" )
    if(ARM_TARGET_ARCH_ABI STREQUAL "armv8")
      set(arm_math_name "math_arm.armlinux.armv8")
    elseif(ARM_TARGET_ARCH_ABI STREQUAL "armv7")
      set(arm_math_name "math_arm.armlinux.armv7")
    endif()
  endif()

  if(ARM_TARGET_LANG STREQUAL "clang")
    set(arm_math_name "${arm_math_name}.clang")
  else()
    set(arm_math_name "${arm_math_name}.gcc")
  endif()

  find_library(math_arm_file ${arm_math_name} ${ARM_MATH_LIB_DIR} NO_DEFAULT_PATH)
  if(math_arm_file)
    add_library(math_arm STATIC IMPORTED GLOBAL)
    set_property(TARGET math_arm PROPERTY IMPORTED_LOCATION ${math_arm_file})
    message(STATUS "ARM math library imported: ${math_arm_file}")
    set(HAS_ARM_MATH_LIB_DIR ON)
  else()
    message(WARNING "Can not find arm math library ${arm_math_name} in ${ARM_MATH_LIB_DIR}")
  endif()
endif()

file(GLOB FP16_MATH_SRC ${CMAKE_CURRENT_SOURCE_DIR}/fp16/*.cc)
file(GLOB COMMON_MATH_SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.cc)
if(LITE_WITH_ARM82_FP16 AND ${ANDROID_NDK_MAJOR} AND ${ANDROID_NDK_MAJOR} GREATER "17")
  add_library(arm_math OBJECT ${COMMON_MATH_SRC} ${FP16_MATH_SRC})
else()
  add_library(arm_math OBJECT ${COMMON_MATH_SRC})
endif()
