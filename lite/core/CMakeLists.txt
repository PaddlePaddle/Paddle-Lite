# source code for unit tests
FILE(GLOB_RECURSE UNIT_TEST_SRC ${CMAKE_CURRENT_SOURCE_DIR}/*test*)

# profiler source code
FILE(GLOB_RECURSE PROFILE_SRC ${CMAKE_CURRENT_SOURCE_DIR}/profile/*.cc)
LIST(REMOVE_ITEM PROFILE_SRC ${UNIT_TEST_SRC})

# model defination source code
FILE(GLOB_RECURSE MODEL_SRC ${CMAKE_CURRENT_SOURCE_DIR}/model/*.cc)
LIST(REMOVE_ITEM MODEL_SRC ${UNIT_TEST_SRC})

# tester source code
FILE(GLOB_RECURSE TESTER_SRC ${CMAKE_CURRENT_SOURCE_DIR}/test/*.cc)
FILE(GLOB_RECURSE TESTER_UNIT_TEST_SRC ${CMAKE_CURRENT_SOURCE_DIR}/test/*test.cc)
LIST(REMOVE_ITEM TESTER_SRC ${TESTER_UNIT_TEST_SRC})

# optimizer source code
FILE(GLOB_RECURSE OPTIMIZER_SRC ${CMAKE_CURRENT_SOURCE_DIR}/optimizer/*.cc)
LIST(REMOVE_ITEM OPTIMIZER_SRC ${UNIT_TEST_SRC})

# subgraph source code
FILE(GLOB_RECURSE SUBGRAPH_SRC ${CMAKE_CURRENT_SOURCE_DIR}/subgraph/*.cc)
LIST(REMOVE_ITEM SUBGRAPH_SRC ${UNIT_TEST_SRC})

# common code for core module
FILE(GLOB CORE_BASE_SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.cc)
LIST(REMOVE_ITEM CORE_BASE_SRC ${UNIT_TEST_SRC})

set (tensor_extra_deps "")
if (LITE_WITH_FPGA)
  set (tensor_extra_deps lite_tensor_fpga)
endif ()


set(CORE_SRC ${CORE_BASE_SRC} ${MODEL_SRC})
set(CORE_DEPS "")

if (LITE_WITH_FPGA)
  set (CORE_DEPS ${CORE_DEPS} lite_tensor_fpga)
endif ()

if(WITH_TESTING)
  lite_cc_library(core_tester SRCS ${TESTER_SRC} DEPS gtest gflags)
  set(CORE_DEPS ${CORE_DEPS} gtest gflags)
endif()

if(NOT LITE_ON_TINY_PUBLISH)
  proto_library(framework_proto SRCS framework.proto)
  set(CORE_SRC ${CORE_SRC} ${OPTIMIZER_SRC})
  set(CORE_DEPS ${CORE_DEPS} framework_proto)
endif()

if (LITE_WITH_PROFILE)
  set(CORE_SRC ${CORE_SRC} ${PROFILE_SRC})
  if (NOT IOS)
    set(CORE_DEPS ${CORE_DEPS} gflags)
  endif()
endif()

if (LITE_WITH_NPU OR LITE_WITH_XTCL OR LITE_WITH_BM OR LITE_WITH_RKNPU OR LITE_WITH_MLU OR LITE_WITH_NNADAPTER)
  set(CORE_SRC  ${CORE_SRC} ${SUBGRAPH_SRC})
endif ()

lite_cc_library(core SRCS ${CORE_SRC}
  DEPS target_wrapper_host fbs_headers ${CORE_DEPS}
  X86_DEPS eigen3
  CUDA_DEPS target_wrapper_cuda cuda_context
  XPU_DEPS target_wrapper_xpu
  CL_DEPS cl_target_wrapper cl_context
  METAL_DEPS metal_target_wrapper
  FPGA_DEPS fpga_target_wrapper
  INTEL_FPGA_DEPS intel_fpga_target_wrapper
  BM_DEPS target_wrapper_bm
  MLU_DEPS target_wrapper_mlu
  NNADAPTER_DEPS nnadapter_wrapper)

#-------------------------------------------- GET CODE META INFO ------------------------------------------
if (LITE_WITH_CODE_META_INFO)
  execute_process (
    COMMAND git describe --tags --exact-match
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE PADDLE_LITE_TAG
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )

  execute_process (
    COMMAND git rev-parse --abbrev-ref HEAD
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE PADDLE_LITE_BRANCH
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )

  execute_process (
    COMMAND git log -1 --format=%h
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE PADDLE_LITE_COMMIT
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )

  message (STATUS "tag: ${PADDLE_LITE_TAG}")
  message (STATUS "branch: ${PADDLE_LITE_BRANCH}")
  message (STATUS "commit: ${PADDLE_LITE_COMMIT}")
endif ()
configure_file (version.h.in version.h)
#----------------------------------------------- NOT CHANGE -----------------------------------------------




add_subdirectory(optimizer)
add_subdirectory(profile)
add_subdirectory(test)


# for mobile, unnecessary to compile the following testings.
if (LITE_WITH_LIGHT_WEIGHT_FRAMEWORK)
  return ()
endif ()

lite_cc_test (test_scope SRCS scope_test.cc)
lite_cc_test (test_kernel SRCS kernel_test.cc)
lite_cc_test (test_op SRCS op_lite_test.cc)
lite_cc_test (test_tensor SRCS lite_tensor_test.cc)
lite_cc_test (test_type_system SRCS type_system_test.cc)
lite_cc_test (test_types SRCS types_test.cc)
lite_cc_test (test_memory SRCS memory_test.cc)
lite_cc_test (test_context SRCS context_test.cc)
