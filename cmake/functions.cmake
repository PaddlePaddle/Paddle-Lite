set(ops_src_list "${PADDLE_BINARY_DIR}/ops_src_list.txt")
set(OPS_SRC CACHE INTERNAL "")
file(WRITE ${ops_src_list} "") # clean
if(LITE_BUILD_TAILOR)
  set(tailored_ops_list_path "${LITE_OPTMODEL_DIR}/.tailored_ops_source_list")
  file(STRINGS ${tailored_ops_list_path} tailored_ops_list)
endif()

# add an operator
# level: one of (basic, extra, train)
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
      set(__lite_cc_files ${__lite_cc_files} "${CMAKE_CURRENT_SOURCE_DIR}/${src}" CACHE INTERNAL "")
      set(OPS_SRC ${OPS_SRC} "${CMAKE_CURRENT_SOURCE_DIR}/${src}" CACHE INTERNAL "source")
    endforeach()
endfunction()


set(kernels_src_list "${PADDLE_BINARY_DIR}/kernels_src_list.txt")
file(WRITE ${kernels_src_list} "") # clean

# file to record faked kernels for opt python lib
set(fake_kernels_src_list "${PADDLE_BINARY_DIR}/fake_kernels_src_list.txt")
file(WRITE ${fake_kernels_src_list} "") # clean

# add a kernel for some specific device
set(IS_FAKED_KERNEL false CACHE INTERNAL "judget faked kernel")
set(cuda_kernels CACHE INTERNAL "cuda kernels")
# device: one of (Host, ARM, X86, NPU, MLU, HUAWEI_ASCEND_NPU, APU, FPGA, OPENCL, CUDA, BM, RKNPU IMAGINATION_NNA)
# level: one of (basic, extra)
function(add_kernel TARGET device level)
    set(options "")
    set(oneValueArgs "")
    set(multiValueArgs SRCS)
    cmake_parse_arguments(args "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})


    if (("${level}" STREQUAL "extra" AND (NOT LITE_BUILD_EXTRA)) OR ("${level}" STREQUAL "train" AND (NOT LITE_WITH_TRAIN)))
        return()
    endif()

    # apppend faked kernels into fake kernels source list.(this is useful in opt tool)
    if(${IS_FAKED_KERNEL})
      foreach(src ${args_SRCS})
        file(APPEND ${fake_kernels_src_list} "${CMAKE_CURRENT_SOURCE_DIR}/${src}\n")
      endforeach()
      return()
    endif()

    # strp lib according to useful kernel names.
    if(LITE_BUILD_TAILOR)
      set(dst_file "")
      foreach(src ${args_SRCS})
        string(TOLOWER "${device}" device_name) # ARM => arm, Host => host
        get_filename_component(filename ${src} NAME_WE) # conv_compute.cc => conv_compute
        set(kernel_tailor_src_dir "${PADDLE_BINARY_DIR}/kernel_tailor_src_dir")
        set(suffix "for_strip")
        set(src_file "${kernel_tailor_src_dir}/${filename}_${device_name}_${suffix}.cc") # conv_compute_arm.cc
        if("${device}" STREQUAL "METAL")
          set(src_file "${kernel_tailor_src_dir}/${filename}_${device_name}_${suffix}.mm") # conv_compute_apple_metal_for_strip.mm
        endif()
        if(NOT EXISTS ${src_file})
          return()
        endif()
        set(dst_file ${dst_file} "${src_file}")
      endforeach()
      file(APPEND ${kernels_src_list} "${dst_file}\n")
      set(KERNELS_SRC ${KERNELS_SRC} "${dst_file}" CACHE INTERNAL "kernels source")
      set(__lite_cc_files ${__lite_cc_files} ${dst_file} CACHE INTERNAL "")
      return()
    endif()

    if ("${device}" STREQUAL "CUDA")
        if (NOT LITE_WITH_CUDA)
            foreach(src ${args_SRCS})
                file(APPEND ${fake_kernels_src_list} "${CMAKE_CURRENT_SOURCE_DIR}/${src}\n")
            endforeach()
            return()
        endif()
        set(cuda_kernels "${cuda_kernels};${TARGET}" CACHE INTERNAL "")
        foreach(src ${args_SRCS})
          file(APPEND ${kernels_src_list} "${CMAKE_CURRENT_SOURCE_DIR}/${src}\n")
        endforeach()
        nv_library(${TARGET} SRCS ${args_SRCS})
        return()
    endif()

    # compile actual kernels
    foreach(src ${args_SRCS})
        file(APPEND ${kernels_src_list} "${CMAKE_CURRENT_SOURCE_DIR}/${src}\n")
        set(KERNELS_SRC ${KERNELS_SRC} "${CMAKE_CURRENT_SOURCE_DIR}/${src}" CACHE INTERNAL "kernels source")
        set(__lite_cc_files ${__lite_cc_files} "${CMAKE_CURRENT_SOURCE_DIR}/${src}" CACHE INTERNAL "")
    endforeach()

endfunction()

# Add a unit-test name to file for latter offline manual test.
set(offline_test_registry_file "${PADDLE_BINARY_DIR}/lite_tests.txt")
file(WRITE ${offline_test_registry_file} "") # clean

function(lite_cc_test TARGET)
  if(NOT WITH_TESTING)
      return()
  endif()
  set(options "")
  set(oneValueArgs "")

  set(multiValueArgs SRCS DEPS X86_DEPS CUDA_DEPS CL_DEPS METAL_DEPS ARM_DEPS FPGA_DEPS INTEL_FPGA_DEPS BM_DEPS
        IMAGINATION_NNA_DEPS RKNPU_DEPS NPU_DEPS XPU_DEPS MLU_DEPS HUAWEI_ASCEND_NPU_DEPS APU_DEPS NNADAPTER_DEPS PROFILE_DEPS
        LIGHT_DEPS HVY_DEPS EXCLUDE_COMPILE_DEPS CV_DEPS
        ARGS
        COMPILE_LEVEL # (basic|extra)
  )
  cmake_parse_arguments(args "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  set(deps "")
  lite_deps(deps
            DEPS ${args_DEPS}
            X86_DEPS ${args_X86_DEPS}
            CUDA_DEPS ${args_CUDA_DEPS}
            CL_DEPS ${args_CL_DEPS}
            METAL_DEPS ${args_METAL_DEPS}
            ARM_DEPS ${args_ARM_DEPS}
            FPGA_DEPS ${args_FPGA_DEPS}
            INTEL_FPGA_DEPS ${args_INTEL_FPGA_DEPS}
            NPU_DEPS ${args_NPU_DEPS}
            APU_DEPS ${args_APU_DEPS}
            XPU_DEPS ${args_XPU_DEPS}
            RKNPU_DEPS ${args_RKNPU_DEPS}
            BM_DEPS ${args_BM_DEPS}
            IMAGINATION_NNA_DEPS ${args_IMAGINATION_NNA_DEPS}
            NNADAPTER_DEPS ${args_NNADAPTER_DEPS}
            PROFILE_DEPS ${args_PROFILE_DEPS}
            LIGHT_DEPS ${args_LIGHT_DEPS}
            HVY_DEPS ${args_HVY_DEPS}
            CV_DEPS ${args_CV_DEPS}
            MLU_DEPS ${args_MLU_DEPS}
            HUAWEI_ASCEND_NPU_DEPS ${args_HUAWEI_ASCEND_NPU_DEPS}
            )
  if(LITE_WITH_ARM)
    cc_binary(${TARGET} SRCS ${args_SRCS} DEPS ${deps} core_tester gflags gtest)
  else()
    cc_binary(${TARGET} SRCS ${args_SRCS} DEPS ${deps} core_tester gflags gtest glog)
  endif()
  file(APPEND ${offline_test_registry_file} "${TARGET}\n")
  add_dependencies(${TARGET} bundle_full_api)
  if(NOT WIN32)
    target_link_libraries(${TARGET} ${PADDLE_BINARY_DIR}/libpaddle_api_full_bundled.a)
  else()
    target_link_libraries(${TARGET} ${PADDLE_BINARY_DIR}/lite/api/${CMAKE_BUILD_TYPE}/libpaddle_api_full_bundled.lib)
  endif()
  # windows
  if(NOT WIN32)
    target_compile_options(${TARGET} BEFORE PRIVATE -Wno-ignored-qualifiers)
  endif()
  # collect targets need to compile for lite
  if (NOT args_EXCLUDE_COMPILE_DEPS)
      add_dependencies(lite_compile_deps ${TARGET})
  endif()

  # link to dynamic runtime lib
  if(LITE_WITH_XPU)
      target_link_libraries(${TARGET} ${xpu_builder_libs} ${xpu_runtime_libs})
  endif()
  if(LITE_WITH_NPU)
      target_link_libraries(${TARGET} ${npu_builder_libs} ${npu_runtime_libs})
  endif()
  if(LITE_WITH_CUDA)
      get_property(cuda_deps GLOBAL PROPERTY CUDA_MODULES)
      target_link_libraries(${TARGET} ${cuda_deps})
  endif()

  set(LINK_FLAGS "-Wl,--version-script ${PADDLE_SOURCE_DIR}/lite/core/lite.map")
  set_target_properties(${TARGET} PROPERTIES LINK_FLAGS "${LINK_FLAGS}")
  common_link(${TARGET})
  add_test(NAME ${TARGET}
          COMMAND ${TARGET} ${args_ARGS}
          WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
  # No unit test should exceed 10 minutes.
  set_tests_properties(${TARGET} PROPERTIES TIMEOUT 1200)

endfunction()

# ----------------------------------------------------------------------------
# section: Provides an paddle lite config option macro
# usage：  lite_option(var "help string to describe the var" [if or IF (condition)])
# ----------------------------------------------------------------------------
macro(lite_option variable description value)
    set(__value ${value})
    set(__condition "")
    set(__varname "__value")
    foreach(arg ${ARGN})
        if(arg STREQUAL "IF" OR arg STREQUAL "if")
            set(__varname "__condition")
        else()
            list(APPEND ${__varname} ${arg})
        endif()
    endforeach()
    unset(__varname)
    if(__condition STREQUAL "")
        set(__condition 2 GREATER 1)
    endif()

    if(${__condition})
        if(__value MATCHES ";")
            if(${__value})
                option(${variable} "${description}" ON)
            else()
                option(${variable} "${description}" OFF)
            endif()
        elseif(DEFINED ${__value})
            if(${__value})
                option(${variable} "${description}" ON)
            else()
                option(${variable} "${description}" OFF)
            endif()
        else()
             option(${variable} "${description}" ${__value})
        endif()
    else()
        unset(${variable} CACHE)
    endif()
    unset(__condition)
    unset(__value)
endmacro()
