# NOTE we leave the add_kernel not protected by LITE_WITH_LIGHT_WEIGHT_FRAMEWORK so that all the kernels will be registered
# to the model_optimize_tool.
if((NOT LITE_ON_MODEL_OPTIMIZE_TOOL) AND (NOT LITE_WITH_PYTHON) AND (NOT (LITE_WITH_LIGHT_WEIGHT_FRAMEWORK AND LITE_WITH_ARM)))
    return()
endif()

message(STATUS "compile with lite ARM TrustZone kernels")

find_library(libtee_compute_file libtee_compute.a ${PADDLE_SOURCE_DIR}/lite/kernels/arm_trustzone/ NO_DEFAULT_PATH)
set(libtee_compute_file ${PADDLE_SOURCE_DIR}/lite/kernels/arm_trustzone/libtee_compute.a)
if(libtee_compute_file)
  add_library(tee_compute STATIC IMPORTED GLOBAL)
  set_property(TARGET tee_compute PROPERTY IMPORTED_LOCATION ${libtee_compute_file})
  message(STATUS "ARM TrustZone compute library imported: ${libtee_compute_file}")
else()
  message(WARNING "Cannot find ARM TrustZone compute library")
endif()

add_kernel(softmax_compute_arm_trustzone ARM_TRUSTZONE basic SRCS softmax_compute.cc tee.cc DEPS ${lite_kernel_deps} math_arm tee_compute)
add_kernel(fc_compute_arm_trustzone ARM_TRUSTZONE basic SRCS fc_compute.cc tee.cc DEPS ${lite_kernel_deps} math_arm_trustzone tee_compute)
lite_cc_test(test_softmax_compute_arm_trustzone SRCS softmax_compute_test.cc DEPS softmax_compute_arm_trustzone)
