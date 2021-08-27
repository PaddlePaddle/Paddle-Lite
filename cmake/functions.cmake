set(ops_src_list "${CMAKE_BINARY_DIR}/ops_src_list.txt")
set(OPS_SRC CACHE INTERNAL "")
file(WRITE ${ops_src_list} "") # clean
if(LITE_BUILD_TAILOR)
  set(tailored_ops_list_path "${LITE_OPTMODEL_DIR}/.tailored_ops_source_list")
  file(STRINGS ${tailored_ops_list_path} tailored_ops_list)
endif()

add_library(ops STATIC "${CMAKE_SOURCE_DIR}/lite/operators/op_params.cc")
add_dependencies(ops utils)
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
      target_sources(ops PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/${src}")
    endforeach()
endfunction()


set(kernels_src_list "${CMAKE_BINARY_DIR}/kernels_src_list.txt")
file(WRITE ${kernels_src_list} "") # clean

# file to record faked kernels for opt python lib
set(fake_kernels_src_list "${CMAKE_BINARY_DIR}/fake_kernels_src_list.txt")
file(WRITE ${fake_kernels_src_list} "") # clean

if(LITE_BUILD_TAILOR)
  set(tailored_kernels_list_path "${LITE_OPTMODEL_DIR}/.tailored_kernels_source_list")
  file(STRINGS ${tailored_kernels_list_path} tailored_kernels_list)
endif()
# add a kernel for some specific device
set(IS_FAKED_KERNEL false)
add_library(kernels STATIC "${CMAKE_SOURCE_DIR}/lite/kernels/host/feed_compute.cc" "${CMAKE_SOURCE_DIR}/lite/kernels/host/fetch_compute.cc")
add_dependencies(kernels utils)
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

    # apppend faked kernels into fake kernels source list.
    if(${IS_FAKED_KERNE})
      foreach(src ${args_SRCS})
        file(APPEND ${fake_kernels_src_list} "${CMAKE_CURRENT_SOURCE_DIR}/${src}\n")
      endforeach()
      return()
    endif()
    # compile actual kernels
    foreach(src ${args_SRCS})
        file(APPEND ${kernels_src_list} "${CMAKE_CURRENT_SOURCE_DIR}/${src}\n")
        set(KERNELS_SRC ${KERNELS_SRC} "${CMAKE_CURRENT_SOURCE_DIR}/${src}" CACHE INTERNAL "kernels source")
        target_sources(kernels "${CMAKE_CURRENT_SOURCE_DIR}/${src}")
        set(__lite_cc_files ${__lite_cc_files} "${CMAKE_CURRENT_SOURCE_DIR}/${src}" CACHE INTERNAL "")
    endforeach()

endfunction()
