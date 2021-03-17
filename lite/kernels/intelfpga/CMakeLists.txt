if ((NOT LITE_ON_MODEL_OPTIMIZE_TOOL) AND (NOT LITE_WITH_PYTHON) AND (NOT LITE_WITH_INTELFPGA))
    return()
endif()

set(intelfpga_deps intelfpga_target_wrapper kernel_intelfpga)

add_kernel(conv_depthwise_intelfpga INTELFPGA basic SRCS conv_depthwise.cc DEPS ${intelfpga_deps})
add_kernel(conv_gemmlike_intelfpga INTELFPGA basic SRCS conv_gemmlike.cc DEPS ${intelfpga_deps})
add_kernel(conv_compute_intelfpga INTELFPGA basic SRCS conv_compute.cc DEPS ${intelfpga_deps} conv_depthwise_intelfpga conv_gemmlike_intelfpga)
