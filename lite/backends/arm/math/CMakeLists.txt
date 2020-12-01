if(NOT (LITE_WITH_LIGHT_WEIGHT_FRAMEWORK AND LITE_WITH_ARM))
    return()
endif()

if(NOT (LITE_WITH_LIGHT_WEIGHT_FRAMEWORK AND LITE_WITH_ARM))
  return()
endif()

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


if (NOT HAS_ARM_MATH_LIB_DIR)
  # TODO(xxx): seperate them and do not deps proto, eigen3
  cc_library(math_arm SRCS
      funcs.cc
      packed_sgemm.cc
      packed_sgemm_c4.cc
      sgemm.cc
      gemm_prepacked_int8.cc
      gemm_s8.cc
      sgemv.cc
      gemv_arm_int8.cc
      conv3x3s1_direct_fp32.cc
      conv3x3s2_direct_fp32.cc
      conv3x3s1p01_depthwise_fp32_relu.cc
      conv3x3s2p01_depthwise_fp32_relu.cc
      conv3x3s1p01_depthwise_fp32.cc
      conv3x3s2p01_depthwise_fp32.cc
      conv3x3s1px_depthwise_fp32.cc
      conv3x3s2px_depthwise_fp32.cc
      conv3x3s1_direct_int8.cc
      conv3x3s2_direct_int8.cc
      conv3x3s1_depthwise_int8.cc
      conv3x3s2_depthwise_int8.cc
      conv5x5s1_depthwise_int8.cc
      conv5x5s1_depthwise_fp32.cc
      conv5x5s2_depthwise_int8.cc
      conv5x5s2_depthwise_fp32.cc
      conv3x3_winograd_fp32_c4.cc
      conv3x3_winograd_int8.cc
      conv_winograd_3x3.cc
      conv_impl.cc
      softmax.cc
      scale.cc
      pooling.cc
      elementwise.cc
      layout.cc
      lrn.cc
      decode_bboxes.cc
      concat.cc
      type_trans.cc
      box_coder.cc
      split.cc
      shuffle_channel.cc
      activation.cc
      yolo_box.cc
      dropout.cc
      power.cc
      interpolate.cc
      argmax.cc
      axpy.cc
      fill_bias_relu.cc
      col_im_transform.cc
      conv_transpose_depthwise.cc
      im2sequence.cc
      prior_box.cc
      sequence_softmax.cc
      norm.cc
      topk.cc
      increment.cc
      pad2d.cc
      negative.cc
      beam_search.cc
      reduce_max.cc
      sequence_pool.cc
      sequence_pool_grad.cc
      sequence_expand.cc
      slice.cc
      reduce_mean.cc
      affine_channel.cc
      anchor_generator.cc
      split_merge_lod_tenosr.cc
      reduce_prod.cc
      reduce_sum.cc
      lstm.cc
      clip.cc
      pixel_shuffle.cc
      scatter.cc
      DEPS ${lite_kernel_deps} context tensor)
endif()
