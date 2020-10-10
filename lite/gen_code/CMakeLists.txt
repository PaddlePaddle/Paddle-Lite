if (LITE_ON_TYNY_PUBLISH)
    return()
endif()

lite_cc_library(gen_code SRCS gen_code.cc
        DEPS program op scope
        cpp_op_desc
        HVY_DEPS operator)
lite_cc_library(paddle_infer_gencode SRCS paddle_infer.cc DEPS program utils)

lite_cc_test(test_gen_code SRCS gen_code_test.cc 
        DEPS gen_code tensor ${host_kernels} ${ops}
        compatible_pb
        model_parser
        X86_DEPS ${x86_kernels}
        ARM_DEPS ${arm_kernels}
        NPU_DEPS ${npu_kernels}
        HUAWEI_ASCEND_NPU_DEPS ${huawei_ascend_npu_kernels}
        RKNPU_DEPS ${rknpu_kernels}
        IMAGINATION_NNA_DEPS ${imagination_nna_kernels}
        XPU_DEPS ${xpu_kernels}
        CL_DEPS ${opencl_kernels}
        FPGA_DEPS ${fpga_kernels}
        EXCLUDE_COMPILE_DEPS "ON"
        ARGS --optimized_model=${LITE_MODEL_DIR}/lite_naive_model_opt SERIAL)

lite_cc_library(__generated_code__
    SRCS ${CMAKE_BINARY_DIR}/lite/gen_code/__generated_code__.cc
    DEPS scope op kernel paddle_infer_gencode
    EXCLUDE_COMPILE_DEPS "ON"
)
if(WITH_TESTING)
    add_dependencies(__generated_code__ test_gen_code)
    add_dependencies(__generated_code__ extern_lite_download_lite_naive_model_tar_gz)
endif(WITH_TESTING)

lite_cc_binary(paddle_code_generator SRCS paddle_code_generator.cc DEPS model_parser gen_code gflags)

# TODO(xxx): fix the gen code bug on ios
if(IOS)
    return()
endif()

lite_cc_test(test_generated_code SRCS generated_code_test.cc DEPS __generated_code__
    ${ops} ${host_kernels}
    X86_DEPS ${x86_kernels}
    ARM_DEPS ${arm_kernels}
    NPU_DEPS ${npu_kernels}
    HUAWEI_ASCEND_NPU_DEPS ${huawei_ascend_npu_kernels}
    RKNPU_DEPS ${rknpu_kernels}
    IMAGINATION_NNA_DEPS ${imagination_nna_kernels}
    XPU_DEPS ${xpu_kernels}
    CL_DEPS ${opencl_kernels}
    FPGA_DEPS ${fpga_kernels}
    EXCLUDE_COMPILE_DEPS "ON"
)
 
