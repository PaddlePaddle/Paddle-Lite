if(NOT LITE_WITH_NPU)
  return ()
endif()

if(NOT DEFINED NPU_DDK_ROOT)
    set(NPU_DDK_ROOT $ENV{NPU_DDK_ROOT})
    if(NOT NPU_DDK_ROOT)
        message(FATAL_ERROR "Must set NPU_DDK_ROOT or env NPU_DDK_ROOT when LITE_WITH_NPU=ON")
    endif()
endif()

message(STATUS "NPU_DDK_ROOT: ${NPU_DDK_ROOT}")
find_path(NPU_DDK_INC NAMES HiAiModelManagerService.h
  PATHS ${NPU_DDK_ROOT}/include NO_DEFAULT_PATH)
if(NOT NPU_DDK_INC)
  message(FATAL_ERROR "Can not find HiAiModelManagerService.h in ${NPU_DDK_ROOT}/include")
endif()

include_directories("${NPU_DDK_ROOT}")

find_library(NPU_DDK_HIAI_FILE NAMES hiai
  PATHS ${NPU_DDK_ROOT}/lib64)

find_library(NPU_DDK_IR_FILE NAMES hiai_ir
  PATHS ${NPU_DDK_ROOT}/lib64)

find_library(NPU_DDK_IR_BUILD_FILE NAMES hiai_ir_build
  PATHS ${NPU_DDK_ROOT}/lib64)

if(NOT NPU_DDK_HIAI_FILE)
  message(FATAL_ERROR "Can not find NPU_DDK_HIAI_FILE in ${NPU_DDK_ROOT}")
else()
  message(STATUS "Found NPU_DDK HIAI Library: ${NPU_DDK_HIAI_FILE}")
  add_library(npu_ddk_hiai SHARED IMPORTED GLOBAL)
  set_property(TARGET npu_ddk_hiai PROPERTY IMPORTED_LOCATION ${NPU_DDK_HIAI_FILE})
endif()

if(NOT NPU_DDK_IR_FILE)
  message(FATAL_ERROR "Can not find NPU_DDK_IR_FILE in ${NPU_DDK_ROOT}")
else()
  message(STATUS "Found NPU_DDK IR Library: ${NPU_DDK_IR_FILE}")
  add_library(npu_ddk_ir SHARED IMPORTED GLOBAL)
  set_property(TARGET npu_ddk_ir PROPERTY IMPORTED_LOCATION ${NPU_DDK_IR_FILE})
endif()

if(NOT NPU_DDK_IR_BUILD_FILE)
  message(FATAL_ERROR "Can not find NPU_DDK_IR_BUILD_FILE in ${NPU_DDK_ROOT}")
else()
  message(STATUS "Found NPU_DDK IR_BUILD Library: ${NPU_DDK_IR_BUILD_FILE}")
  add_library(npu_ddk_ir_build SHARED IMPORTED GLOBAL)
  set_property(TARGET npu_ddk_ir_build PROPERTY IMPORTED_LOCATION ${NPU_DDK_IR_BUILD_FILE})
endif()

set(npu_ddk_libs npu_ddk_hiai npu_ddk_ir npu_ddk_ir_build CACHE INTERNAL "npu ddk libs")

lite_cc_library(npu_helper SRCS npu_helper.cc DEPS ${npu_ddk_libs})
add_subdirectory(bridge)
