if (NOT LITE_WITH_INTEL_FPGA)
    return()
endif()

lite_cc_library(intel_fpga_target_wrapper SRCS target_wrapper.cpp DEPS ${intel_fpga_runtime_libs})
