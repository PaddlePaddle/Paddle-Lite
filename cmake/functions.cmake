set(ops_src_list "${CMAKE_BINARY_DIR}/ops_src_list.txt")
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


set(kernels_src_list "${CMAKE_BINARY_DIR}/kernels_src_list.txt")
file(WRITE ${kernels_src_list} "") # clean

# file to record faked kernels for opt python lib
set(fake_kernels_src_list "${CMAKE_BINARY_DIR}/fake_kernels_src_list.txt")
file(WRITE ${fake_kernels_src_list} "") # clean

# add a kernel for some specific device
set(IS_FAKED_KERNEL false)
# device: one of (Host, ARM, X86, NPU, MLU, HUAWEI_ASCEND_NPU, APU, FPGA, OPENCL, CUDA, BM, RKNPU IMAGINATION_NNA)
# level: one of (basic, extra)
function(add_kernel TARGET device level)
    set(options "")
    set(oneValueArgs "")
    set(multiValueArgs SRCS )
    cmake_parse_arguments(args "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})


    if (("${level}" STREQUAL "extra" AND (NOT LITE_BUILD_EXTRA)) OR ("${level}" STREQUAL "train" AND (NOT LITE_WITH_TRAIN)))
        return()
    endif()

    # apppend faked kernels into fake kernels source list.(this is useful in opt tool)
    if(${IS_FAKED_KERNE})
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
        set(kernel_tailor_src_dir "${CMAKE_BINARY_DIR}/kernel_tailor_src_dir")
        set(suffix "for_strip")
        set(dst_file ${dst_file} "${kernel_tailor_src_dir}/${filename}_${device_name}_${suffix}.cc") # conv_compute_arm.cc
        if("${device}" STREQUAL "METAL")
          set(dst_file ${dst_file} "${kernel_tailor_src_dir}/${filename}_${device_name}_${suffix}.mm") # conv_compute_apple_metal_for_strip.mm
        endif()
        if(NOT EXISTS ${dst_file})
          return()
        endif()
      endforeach()
      target_sources(kernels PUBLIC ${dst_file})
      set(__lite_cc_files ${__lite_cc_files} ${dst_file} CACHE INTERNAL "")
      return()
    endif()

    # compile actual kernels
    foreach(src ${args_SRCS})
        file(APPEND ${kernels_src_list} "${CMAKE_CURRENT_SOURCE_DIR}/${src}\n")
        set(KERNELS_SRC ${KERNELS_SRC} "${CMAKE_CURRENT_SOURCE_DIR}/${src}" CACHE INTERNAL "kernels source")
        set(__lite_cc_files ${__lite_cc_files} "${CMAKE_CURRENT_SOURCE_DIR}/${src}" CACHE INTERNAL "")
    endforeach()

endfunction()
