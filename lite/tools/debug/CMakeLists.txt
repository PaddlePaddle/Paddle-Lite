if(NOT LITE_ON_MODEL_OPTIMIZE_TOOL)
    return()
endif()

lite_cc_library(debug_utils SRCS debug_utils.cc DEPS op_params model_parser)
lite_cc_binary(lite_model_debug_tool SRCS model_debug_tool.cc
    DEPS
    cxx_api
    debug_utils
    target_wrapper_host
    mir_passes
    gflags
    logging
    ${ops} ${host_kernels}
    X86_DEPS ${x86_kernels}
    ARM_DEPS ${arm_kernels}
    NPU_DEPS ${npu_kernels}
    XPU_DEPS ${xpu_kernels}
    FPGA_DEPS ${fpga_kernels}
    CL_DEPS ${opencl_kernels})
