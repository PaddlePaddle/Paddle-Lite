# add a kernel for some specific device
# device: one of (Host, ARM, X86, NPU, MLU, HUAWEI_ASCEND_NPU, APU, FPGA, OPENCL, CUDA, BM, RKNPU IMAGINATION_NNA)
# level: one of (basic, extra)
set(kernels_src_list "${CMAKE_BINARY_DIR}/kernels_src_list.txt")
file(WRITE ${kernels_src_list} "") # clean

# file to record faked kernels for opt python lib
set(fake_kernels_src_list "${CMAKE_BINARY_DIR}/fake_kernels_src_list.txt")
file(WRITE ${fake_kernels_src_list} "") # clean

set(KERNELS_SRC CACHE INTERNAL "kernels")
function(add_kernel TARGET device level)
    set(options "")
    set(oneValueArgs "")
    set(multiValueArgs SRCS DEPS X86_DEPS CUDA_DEPS CL_DEPS METAL_DEPS ARM_DEPS FPGA_DEPS INTEL_FPGA_DEPS BM_DEPS IMAGINATION_NNA_DEPS RKNPU_DEPS NPU_DEPS XPU_DEPS MLU_DEPS HUAWEI_ASCEND_NPU_DEPS APU_DEPS NNADAPTER_DEPS PROFILE_DEPS
        LIGHT_DEPS HVY_DEPS EXCLUDE_COMPILE_DEPS
        ARGS)
    cmake_parse_arguments(args "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if(LITE_BUILD_TAILOR)
      foreach(src ${args_SRCS})
        string(TOLOWER "${device}" device_name) # ARM => arm, Host => host
        get_filename_component(filename ${src} NAME_WE) # conv_compute.cc => conv_compute
        set(kernel_tailor_src_dir "${CMAKE_BINARY_DIR}/kernel_tailor_src_dir")
        set(suffix "for_strip")
        set(dst_file "${kernel_tailor_src_dir}/${filename}_${device_name}_${suffix}.cc") # conv_compute_arm.cc
        if("${device}" STREQUAL "METAL")
          set(dst_file "${kernel_tailor_src_dir}/${filename}_${device_name}_${suffix}.mm") # conv_compute_apple_metal_for_strip.mm
        endif()
        if(NOT EXISTS ${dst_file})
          return()
        endif()
      endforeach()
    endif()

    if ("${level}" STREQUAL "extra" AND (NOT LITE_BUILD_EXTRA))
        return()
    endif()
    if ("${level}" STREQUAL "train" AND (NOT LITE_WITH_TRAIN))
        return()
    endif()

    if (LITE_ON_MODEL_OPTIMIZE_TOOL)
       foreach(src ${args_SRCS})
           file(APPEND ${fake_kernels_src_list} "${CMAKE_CURRENT_SOURCE_DIR}/${src}\n")
       endforeach()
       return()
   else()
       # the source list will collect for paddle_use_kernel.h code generation.
       foreach(src ${args_SRCS})
           file(APPEND ${kernels_src_list} "${CMAKE_CURRENT_SOURCE_DIR}/${src}\n")
           set(KERNELS_SRC ${KERNELS_SRC} "${CMAKE_CURRENT_SOURCE_DIR}/${src}" CACHE INTERNAL "kernel source")
       endforeach() 
   endif()
endfunction()



set(ops_src_list "${CMAKE_BINARY_DIR}/ops_src_list.txt")
set(OPS_SRC CACHE INTERNAL "")
file(WRITE ${ops_src_list} "") # clean
if(LITE_BUILD_TAILOR)
  set(tailored_ops_list_path "${LITE_OPTMODEL_DIR}/.tailored_ops_source_list")
  file(STRINGS ${tailored_ops_list_path} tailored_ops_list)
endif()
# add an operator
# level: one of (basic, extra)
function(add_operator TARGET level)
    set(options "")
    set(oneValueArgs "")
    set(multiValueArgs SRCS)
    cmake_parse_arguments(args "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if ("${level}" STREQUAL "extra" AND (NOT LITE_BUILD_EXTRA))
        return()
    endif()

    if ("${level}" STREQUAL "train" AND (NOT LITE_WITH_TRAIN))
        return()
    endif()

    foreach(src ${args_SRCS})
      if(LITE_BUILD_TAILOR)
        list(FIND tailored_ops_list ${src} _index)
        if (${_index} EQUAL -1)
          return()
        endif()
      endif()
      file(APPEND ${ops_src_list} "${CMAKE_CURRENT_SOURCE_DIR}/${src}\n")
      set(OPS_SRC ${OPS_SRC} "${CMAKE_CURRENT_SOURCE_DIR}/${src}" CACHE INTERNAL "source")
    endforeach()
endfunction()





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
  ${LITE_BUILD_EXTRA}
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
#----------------------------------------------- NOT CHANGE -----------------------------------------------
