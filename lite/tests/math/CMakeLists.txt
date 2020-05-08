if((NOT LITE_WITH_OPENCL AND NOT LITE_WITH_FPGA AND NOT LITE_WITH_MLU) AND (LITE_WITH_X86 OR LITE_WITH_ARM))
    lite_cc_test(sgemm_compute_test SRCS sgemm_compute_test.cc DEPS arena_framework ${arm_kernels} ${lite_ops} ${host_kernels})
    lite_cc_test(sgemv_compute_test SRCS sgemv_compute_test.cc DEPS arena_framework ${arm_kernels} ${lite_ops} ${host_kernels})
    lite_cc_test(sgemm_c4_compute_test SRCS sgemm_c4_compute_test.cc DEPS arena_framework ${arm_kernels} ${lite_ops} ${host_kernels})
    lite_cc_test(gemm_int8_compute_test SRCS gemm_int8_compute_test.cc DEPS arena_framework ${arm_kernels} ${lite_ops} ${host_kernels})
    lite_cc_test(gemv_int8_compute_test SRCS gemv_int8_compute_test.cc DEPS arena_framework ${arm_kernels} ${lite_ops} ${host_kernels})
    lite_cc_test(conv_compute_test SRCS conv_compute_test.cc DEPS arena_framework ${arm_kernels} ${lite_ops} ${host_kernels})
    lite_cc_test(conv_transpose_compute_test SRCS conv_transpose_compute_test.cc DEPS arena_framework ${arm_kernels} ${lite_ops} ${host_kernels})
    lite_cc_test(conv_int8_compute_test SRCS conv_int8_compute_test.cc DEPS arena_framework ${arm_kernels} ${lite_ops} ${host_kernels})
    lite_cc_test(pool_compute_test SRCS pool_compute_test.cc DEPS arena_framework ${arm_kernels} ${lite_ops} ${host_kernels})

    if(LITE_BUILD_EXTRA)
        lite_cc_test(layout_compute_test SRCS layout_compute_test.cc DEPS arena_framework ${arm_kernels} ${lite_ops} ${host_kernels})
    endif()
    

endif()
