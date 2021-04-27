lite_cc_library(core SRCS
     program.cc # program struct
     target_wrapper.cc # basic memory operations: malloc/memcpy/free/memcmp
     memory.cc # memory operations are integrated according to hardware type
     tensor.cc # A unit for data operation
     types.cc  # types for model transformation, useless in tiny_publish
     type_system.cc # type system for op and kernel registry
     context.cc # context designed for different hardware
     device_info.cc # hardware device info
     variable.cc  # variable struct
     scope.cc # scope struct to store runtime variables
     kernel.cc # kernel defination
     op_lite.cc # op defination
     op_registry.cc  # op registry
  DEPS place target_wrapper_host
  X86_DEPS target_wrapper_x86 eigen3
  CUDA_DEPS target_wrapper_cuda cuda_context nvtx_wrapper cuda_type_trans
  XPU_DEPS target_wrapper_xpu
  CL_DEPS cl_target_wrapper cl_context
  FPGA_DEPS fpga_target_wrapper lite_tensor_fpga
  METAL_DEPS metal_target_wrapper
  INTEL_FPGA_DEPS intel_fpga_target_wrapper
  BM_DEPS target_wrapper_bm
  MLU_DEPS target_wrapper_mlu
  PROFILE_DEPS lite_profiler
)

# head files for ops and kernels registry
add_dependencies(core kernel_list_h op_list_h)




# protobuf model data structure defination
if (NOT LITE_ON_TINY_PUBLISH)
    proto_library(framework_proto SRCS framework.proto)
    add_dependencies(core framework_proto)
endif()

# basic struct for unit test
if (WITH_TESTING)
    lite_cc_library(lite_gtest_main SRCS lite_gtest_main.cc DEPS gtest gflags)
endif()

#-------------------------------------------- GET CODE META INFO ------------------------------------------
if (LITE_WITH_CODE_META_INFO)
execute_process(
  COMMAND git describe --tags --exact-match
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  OUTPUT_VARIABLE PADDLE_LITE_TAG
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

execute_process(
  COMMAND git rev-parse --abbrev-ref HEAD
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  OUTPUT_VARIABLE PADDLE_LITE_BRANCH
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

execute_process(
  COMMAND git log -1 --format=%h
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  OUTPUT_VARIABLE PADDLE_LITE_COMMIT
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

message(STATUS "tag: ${PADDLE_LITE_TAG}")
message(STATUS "branch: ${PADDLE_LITE_BRANCH}")
message(STATUS "commit: ${PADDLE_LITE_COMMIT}")
endif()
configure_file(version.h.in version.h)
#----------------------------------------------- NOT CHANGE -----------------------------------------------
# A trick to generate the opencl_kernels_source.cc
#add_custom_command(
#  COMMAND python ${CMAKE_SOURCE_DIR}/lite/tools/cmake_tools/gen_opencl_code.py
#  ${CMAKE_SOURCE_DIR}/lite/backends/opencl/cl_kernel
#  ${CMAKE_BINARY_DIR}/lite/backends/opencl/opencl_kernels_source.cc
#  OUTPUT opencl_kernels_source.cc # not a real path to the output to force it execute every time.
#  )
# A trick to generate the paddle_use_kernels.h
add_custom_command(
  COMMAND python ${CMAKE_SOURCE_DIR}/lite/tools/cmake_tools/parse_kernel_registry.py
  ${kernels_src_list}
  ${CMAKE_SOURCE_DIR}/lite/api/paddle_use_kernels.h
  "${LITE_OPTMODEL_DIR}/.tailored_kernels_list"
  ${LITE_BUILD_TAILOR}
  ${LITE_BUILD_EXTRA}
  ${LITE_WITH_ARM82_FP16}
  OUTPUT kernels.h # not a real path to the output to force it execute every time.
  )
# A trick to generate the paddle_use_ops.h
add_custom_command(
  COMMAND python ${CMAKE_SOURCE_DIR}/lite/tools/cmake_tools/parse_op_registry.py
  ${ops_src_list}
  ${CMAKE_SOURCE_DIR}/lite/api/paddle_use_ops.h
  "${LITE_OPTMODEL_DIR}/.tailored_ops_list"
  ${LITE_BUILD_TAILOR}
  OUTPUT ops.h # not a real path to the output to force it execute every time.
  )
# generate fake kernels for memory_optimize_tool

#-------------------------------opt----------------------------------------------------------------
# tricks to create headfiles for opt
add_custom_command(
  COMMAND python ${CMAKE_SOURCE_DIR}/lite/tools/cmake_tools/create_fake_kernel_registry.py
  ${kernels_src_list}
  ${fake_kernels_src_list}
  ${CMAKE_BINARY_DIR}/all_kernel_faked.cc
  ${CMAKE_BINARY_DIR}/kernel_src_map.h
  OUTPUT all_kernel_faked.cc # not a real path to the output to force it execute every time.
  )
add_custom_target(op_list_h DEPENDS ops.h)
add_custom_target(kernel_list_h DEPENDS kernels.h)
add_custom_target(all_kernel_faked_cc DEPENDS all_kernel_faked.cc)

# create headfile to restore ops info sorted by suppported platforms
add_custom_command(
  COMMAND python ${CMAKE_SOURCE_DIR}/lite/tools/cmake_tools/record_supported_kernel_op.py
  ${kernels_src_list}
  ${fake_kernels_src_list}
  ${ops_src_list}
  ${CMAKE_BINARY_DIR}/supported_kernel_op_info.h
  OUTPUT supported_kernel_op_info.h # not a real path to the output to force it execute every time.
  )
  add_custom_target(supported_kernel_op_info_h DEPENDS supported_kernel_op_info.h)





if (NOT LITE_ON_TINY_PUBLISH)
  lite_cc_library(optimizer SRCS optimizer.cc DEPS mir_pass_manager model_parser core)
  add_subdirectory(mir)
  add_subdirectory(profile)
  add_subdirectory(arena)
endif()

if (LITE_WITH_XCODE)
  add_subdirectory(profile)
endif()

if(LITE_WITH_NPU OR LITE_WITH_XTCL OR LITE_WITH_BM OR LITE_WITH_RKNPU OR LITE_WITH_MLU OR LITE_WITH_APU OR LITE_WITH_HUAWEI_ASCEND_NPU OR LITE_WITH_IMAGINATION_NNA)
    lite_cc_library(subgraph_bridge_registry
      SRCS subgraph_bridge_registry.cc
      DEPS core)
   lite_cc_library(subgraph_engine_base
      SRCS subgraph_engine_base.cc
      DEPS core)
endif()

# for mobile, unnecessary to compile the following testings.
if (LITE_WITH_LIGHT_WEIGHT_FRAMEWORK)
    return()
endif()

lite_cc_test(test_scope SRCS scope_test.cc DEPS ${UNIT_TEST_DEPS})
lite_cc_test(test_kernel SRCS kernel_test.cc DEPS kernel ${UNIT_TEST_DEPS})
lite_cc_test(test_op SRCS op_lite_test.cc DEPS ${UNIT_TEST_DEPS})
lite_cc_test(test_tensor SRCS lite_tensor_test.cc DEPS ${UNIT_TEST_DEPS})
lite_cc_test(test_type_system SRCS type_system_test.cc DEPS type_system ${UNIT_TEST_DEPS})
#lite_cc_test(test_optimizer SRCS optimizer_test.cc DEPS mir_pass_manager program_fake_utils mir_passes optimizer fc_op)
lite_cc_test(test_types SRCS types_test.cc DEPS ${UNIT_TEST_DEPS})
lite_cc_test(test_memory SRCS memory_test.cc DEPS ${UNIT_TEST_DEPS})
lite_cc_test(test_context SRCS context_test.cc DEPS ${UNIT_TEST_DEPS})
