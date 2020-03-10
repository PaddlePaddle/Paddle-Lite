# To make sure the test framework is only actived in TESTING mode.
if(NOT WITH_TESTING)
    return()
endif()

lite_cc_library(arena_framework SRCS framework.cc DEPS program gtest)

if((NOT LITE_WITH_OPENCL) AND (LITE_WITH_X86 OR LITE_WITH_ARM))
  lite_cc_test(test_arena_framework SRCS framework_test.cc DEPS arena_framework ${mlu_kernels} ${bm_kernels} ${npu_kernels} ${xpu_kernels} ${x86_kernels} ${cuda_kernels} ${fpga_kernels} ${arm_kernels} ${lite_ops} ${host_kernels})
endif()
