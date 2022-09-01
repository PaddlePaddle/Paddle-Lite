# NOTE we leave the add_kernel not protected by LITE_WITH_LIGHT_WEIGHT_FRAMEWORK so that all the kernels will be registered
# to the model_optimize_tool.

if(LITE_WITH_ARM_TRUSTZONE AND NOT LITE_ON_MODEL_OPTIMIZE_TOOL)
  set(IS_FAKED_KERNEL false CACHE INTERNAL "")
else()
  return()
endif()

message(STATUS "compile with lite ARM TrustZone kernels")

find_library(libtee_compute_file libtee_compute.a ${PADDLE_SOURCE_DIR}/lite/kernels/arm_trustzone/ NO_DEFAULT_PATH)
set(libtee_compute_file ${PADDLE_SOURCE_DIR}/lite/kernels/arm_trustzone/libtee_compute.a)
if(libtee_compute_file)
  add_library(tee_compute STATIC IMPORTED GLOBAL)
  set_property(TARGET tee_compute PROPERTY IMPORTED_LOCATION ${libtee_compute_file})
  set(lite_kernel_deps ${lite_kernel_deps} tee_compute CACHE INTERNAL "")
  message(STATUS "ARM TrustZone compute library imported: ${libtee_compute_file}")
else()
  message(WARNING "Cannot find ARM TrustZone compute library")
endif()

add_kernel(softmax_compute_arm_trustzone ARM_TRUSTZONE basic SRCS softmax_compute.cc tee.cc)
add_kernel(fc_compute_arm_trustzone ARM_TRUSTZONE basic SRCS fc_compute.cc tee.cc)

lite_cc_test(test_softmax_compute_arm_trustzone SRCS softmax_compute_test.cc)
