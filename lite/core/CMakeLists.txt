if (WITH_TESTING)
    lite_cc_library(lite_gtest_main SRCS lite_gtest_main.cc DEPS gtest gflags)
endif()
lite_cc_library(target_wrapper SRCS target_wrapper.cc
  DEPS target_wrapper_host place
  X86_DEPS target_wrapper_x86
  CUDA_DEPS target_wrapper_cuda
  CL_DEPS cl_target_wrapper
  FPGA_DEPS fpga_target_wrapper)

lite_cc_library(memory SRCS memory.cc DEPS target_wrapper CL_DEPS cl_target_wrapper)
lite_cc_library(tensor SRCS tensor.cc DEPS memory)

proto_library(framework_proto SRCS framework.proto)

if (LITE_WITH_X86)
lite_cc_library(variable SRCS variable.cc DEPS framework_proto)
lite_cc_library(types SRCS types.cc DEPS framework_proto)
else()
lite_cc_library(variable SRCS variable.cc)
lite_cc_library(types SRCS types.cc)
endif()
lite_cc_library(op_registry SRCS op_registry.cc DEPS framework_proto)
lite_cc_library(scope SRCS scope.cc DEPS tensor)
lite_cc_library(cpu_info SRCS cpu_info.cc)

if (LITE_WITH_ARM OR LITE_WITH_FPGA)
lite_cc_library(context SRCS context.cc DEPS tensor any cpu_info CL_DEPS cl_context)
else()
lite_cc_library(context SRCS context.cc DEPS tensor any cpu_info eigen3 CL_DEPS cl_context)
endif()
lite_cc_library(kernel SRCS kernel.cc DEPS context type_system target_wrapper any op_params framework_proto tensor)
lite_cc_library(op SRCS op_lite.cc DEPS scope op_registry target_wrapper kernel
  cpp_op_desc tensor)
lite_cc_library(type_system SRCS type_system.cc DEPS tensor target_wrapper)

lite_cc_library(program SRCS program.cc
    DEPS op kernel model_parser ${ops} ${cpp_wrapper}
    PROFILE_DEPS basic_profiler)
lite_cc_library(optimizer SRCS optimizer.cc DEPS mir_pass_manager model_parser program)

add_subdirectory(mir)
add_subdirectory(profile)
add_subdirectory(arena)

# for mobile, unnecessary to compile the following testings.
if (LITE_WITH_LIGHT_WEIGHT_FRAMEWORK)
    return()
endif()

# lite_cc_library(program_fake_utils SRCS program_fake_utils.cc DEPS mir_ssa_graph
#         scope op_registry proto_desc op
#         ${ops}
#         ${host_kernels}
#         )

lite_cc_test(test_scope SRCS scope_test.cc DEPS scope)
lite_cc_test(test_kernel SRCS kernel_test.cc DEPS kernel target_wrapper any)
lite_cc_test(test_op SRCS op_lite_test.cc DEPS op)
lite_cc_test(test_tensor SRCS lite_tensor_test.cc DEPS tensor)
lite_cc_test(test_type_system SRCS type_system_test.cc DEPS type_system utils)
#lite_cc_test(test_optimizer SRCS optimizer_test.cc DEPS mir_pass_manager program_fake_utils mir_passes optimizer fc_op)
lite_cc_test(test_types SRCS types_test.cc DEPS types)
lite_cc_test(test_memory SRCS memory_test.cc DEPS memory)
lite_cc_test(test_context SRCS context_test.cc DEPS context)
