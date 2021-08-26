set(ops_src_list "${CMAKE_BINARY_DIR}/ops_src_list.txt")
set(OPS_SRC CACHE INTERNAL "")
file(WRITE ${ops_src_list} "") # clean
if(LITE_BUILD_TAILOR)
  set(tailored_ops_list_path "${LITE_OPTMODEL_DIR}/.tailored_ops_source_list")
  file(STRINGS ${tailored_ops_list_path} tailored_ops_list)
endif()

add_library(ops STATIC "${CMAKE_SOURCE_DIR}/lite/operators/op_params.cc")
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
