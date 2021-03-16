if ((NOT LITE_ON_MODEL_OPTIMIZE_TOOL) AND (NOT LITE_WITH_PYTHON) AND (NOT LITE_WITH_MMA))
    return()
endif()

set(mma_deps mma_target_wrapper kernel_mma)

add_kernel(conv_depthwise_mma MMA basic SRCS conv_depthwise.cc DEPS ${mma_deps})
add_kernel(conv_gemmlike_mma MMA basic SRCS conv_gemmlike.cc DEPS ${mma_deps})
add_kernel(conv_compute_mma MMA basic SRCS conv_compute.cc DEPS ${mma_deps} conv_depthwise_mma conv_gemmlike_mma)
