set(LITE_URL "http://paddle-inference-dist.bj.bcebos.com" CACHE STRING "inference download url")

function(lite_download_and_uncompress INSTALL_DIR URL FILENAME)
    message(STATUS "Download inference test stuff from ${URL}/${FILENAME}")
    string(REGEX REPLACE "[-%.]" "_" FILENAME_EX ${FILENAME})
    set(EXTERNAL_PROJECT_NAME "extern_lite_download_${FILENAME_EX}")
    set(UNPACK_DIR "${INSTALL_DIR}/src/${EXTERNAL_PROJECT_NAME}")
    ExternalProject_Add(
            ${EXTERNAL_PROJECT_NAME}
            ${EXTERNAL_PROJECT_LOG_ARGS}
            PREFIX                ${INSTALL_DIR}
            DOWNLOAD_COMMAND      wget --no-check-certificate -q -O ${INSTALL_DIR}/${FILENAME} ${URL}/${FILENAME} && ${CMAKE_COMMAND} -E tar xzf ${INSTALL_DIR}/${FILENAME}
            DOWNLOAD_DIR          ${INSTALL_DIR}
            DOWNLOAD_NO_PROGRESS  1
            CONFIGURE_COMMAND     ""
            BUILD_COMMAND         ""
            UPDATE_COMMAND        ""
            INSTALL_COMMAND       ""
    )
endfunction()

function (lite_deps TARGET)
  set(options "")
  set(oneValueArgs "")
  set(multiValueArgs DEPS X86_DEPS CUDA_DEPS ARM_DEPS PROFILE_DEPS LIGHT_DEPS HVY_DEPS CL_DEPS FPGA_DEPS NPU_DEPS ARGS)
  cmake_parse_arguments(lite_deps "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  set(deps ${lite_deps_DEPS})

  if(LITE_WITH_X86)
    foreach(var ${lite_deps_X86_DEPS})
      set(deps ${deps} ${var})
    endforeach(var)
  endif()

  if(LITE_WITH_CUDA)
    foreach(var ${lite_deps_CUDA_DEPS})
      set(deps ${deps} ${var})
    endforeach(var)
  endif()

  if(LITE_WITH_ARM)
    foreach(var ${lite_deps_ARM_DEPS})
      set(deps ${deps} ${var})
    endforeach(var)
  endif()

  if(LITE_WITH_PROFILE)
    foreach(var ${lite_deps_PROFILE_DEPS})
      set(deps ${deps} ${var})
    endforeach(var)
  endif()

  if(LITE_WITH_LIGHT_WEIGHT_FRAMEWORK)
    foreach(var ${lite_deps_LIGHT_DEPS})
      set(deps ${deps} ${var})
    endforeach(var)
  endif()



  if (NOT LITE_WITH_LIGHT_WEIGHT_FRAMEWORK)
    foreach(var ${lite_deps_HVY_DEPS})
      set(deps ${deps} ${var})
    endforeach(var)
  endif()

  if (LITE_WITH_OPENCL)
    foreach(var ${lite_deps_CL_DEPS})
      set(deps ${deps} ${var})
    endforeach(var)
  endif()

  if (LITE_WITH_FPGA)
    foreach(var ${lite_deps_FPGA_DEPS})
      set(deps ${deps} ${var})
    endforeach(var)
  endif()

  if (LITE_WITH_NPU)
    foreach(var ${lite_deps_NPU_DEPS})
      set(deps ${deps} ${var})
    endforeach(var)
  endif()

  set(${TARGET} ${deps} PARENT_SCOPE)
endfunction()


# A fake target to include all the libraries and tests the lite module depends.
add_custom_target(lite_compile_deps COMMAND echo 1)

# Add names for lite libraries for latter compile. We use this name list to avoid compiling
# the whole fluid project to accelerate the compile speed.
set(offline_lib_registry_file "${CMAKE_BINARY_DIR}/lite_libs.txt")
file(WRITE ${offline_lib_registry_file} "") # clean

# cc_library with branch support.
# The branches:
#  X86_DEPS: works only when LITE_WITH_X86 is ON.
#  CUDA_DEPS:     LITE_WITH_CUDA
#  ARM_DEPS:      LITE_WITH_ARM
#  PROFILE_DEPS:  LITE_WITH_PROFILE
#  LIGHT_DEPS:    LITE_WITH_LIGHT_WEIGHT_FRAMEWORK
#  HVY_DEPS:      NOT LITE_WITH_LIGHT_WEIGHT_FRAMEWORK
#  EXCLUDE_COMPILE_DEPS: TARGET will not be included in lite_compile_deps if this is not None
function(lite_cc_library TARGET)
    set(options SHARED shared STATIC static MODULE module)
    set(oneValueArgs "")
    set(multiValueArgs SRCS DEPS X86_DEPS CUDA_DEPS CL_DEPS NPU_DEPS ARM_DEPS FPGA_DEPS PROFILE_DEPS LIGHT_DEPS
      HVY_DEPS EXCLUDE_COMPILE_DEPS ARGS)
    cmake_parse_arguments(args "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    set(deps "")
    lite_deps(deps
            DEPS ${args_DEPS}
            X86_DEPS ${args_X86_DEPS}
            CUDA_DEPS ${args_CUDA_DEPS}
            CL_DEPS ${args_CL_DEPS}
            NPU_DEPS ${args_NPU_DEPS}
            ARM_DEPS ${args_ARM_DEPS}
            FPGA_DEPS ${args_FPGA_DEPS}
            PROFILE_DEPS ${args_PROFILE_DEPS}
            LIGHT_DEPS ${args_LIGHT_DEPS}
            HVY_DEPS ${args_HVY_DEPS}
            )

    if (args_SHARED OR ARGS_shared)
        cc_library(${TARGET} SRCS ${args_SRCS} DEPS ${deps} ${args_DEPS} SHARED)
    elseif (args_MODULE OR ARGS_module)
        add_library(${TARGET} MODULE ${args_SRCS})
        add_dependencies(${TARGET} ${deps} ${args_DEPS})
    else()
        cc_library(${TARGET} SRCS ${args_SRCS} DEPS ${deps} ${args_DEPS})
    endif()
    target_compile_options(${TARGET} BEFORE PRIVATE -Wno-ignored-qualifiers)

    # collect targets need to compile for lite
    if (args_SRCS AND NOT args_EXCLUDE_COMPILE_DEPS)
        add_dependencies(lite_compile_deps ${TARGET})
    endif()

    # register a library name.
    file(APPEND ${offline_lib_registry_file} "${TARGET}\n")
endfunction()

function(lite_cc_binary TARGET)
    set(options "")
    set(oneValueArgs "")
    set(multiValueArgs SRCS DEPS X86_DEPS CUDA_DEPS CL_DEPS ARM_DEPS FPGA_DEPS PROFILE_DEPS
      LIGHT_DEPS HVY_DEPS EXCLUDE_COMPILE_DEPS ARGS)
    cmake_parse_arguments(args "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    set(deps "")
    lite_deps(deps
            DEPS ${args_DEPS}
            X86_DEPS ${args_X86_DEPS}
            CUDA_DEPS ${args_CUDA_DEPS}
            CL_DEPS ${args_CL_DEPS}
            ARM_DEPS ${args_ARM_DEPS}
            FPGA_DEPS ${args_FPGA_DEPS}
            PROFILE_DEPS ${args_PROFILE_DEPS}
            LIGHT_DEPS ${args_LIGHT_DEPS}
            HVY_DEPS ${args_HVY_DEPS}
            )
    cc_binary(${TARGET} SRCS ${args_SRCS} DEPS ${deps} ${args_DEPS})
    target_compile_options(${TARGET} BEFORE PRIVATE -Wno-ignored-qualifiers)
    # collect targets need to compile for lite
    if (NOT args_EXCLUDE_COMPILE_DEPS)
        add_dependencies(lite_compile_deps ${TARGET})
    endif()
endfunction()

# Add a unit-test name to file for latter offline manual test.
set(offline_test_registry_file "${CMAKE_BINARY_DIR}/lite_tests.txt")
file(WRITE ${offline_test_registry_file} "") # clean
# Test lite modules.

function(lite_cc_test TARGET)
    if(NOT WITH_TESTING)
        return()
    endif()
    set(options "")
    set(oneValueArgs "")
    set(multiValueArgs SRCS DEPS X86_DEPS CUDA_DEPS CL_DEPS ARM_DEPS FPGA_DEPS PROFILE_DEPS
        LIGHT_DEPS HVY_DEPS EXCLUDE_COMPILE_DEPS
        ARGS
        COMPILE_LEVEL # (basic|extra)
        )
    cmake_parse_arguments(args "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if (args_COMPILE_LEVEL STREQUAL "extra" AND (NOT LITE_BUILD_EXTRA))
      MESSAGE(STATUS "Ignore test ${TARGET} due to compile level ${args_COMPILE_LEVEL}")
      return()
    endif()

    set(deps "")
    lite_deps(deps
              DEPS ${args_DEPS}
              X86_DEPS ${args_X86_DEPS}
              CUDA_DEPS ${args_CUDA_DEPS}
              CL_DEPS ${args_CL_DEPS}
              ARM_DEPS ${args_ARM_DEPS}
              FPGA_DEPS ${args_FPGA_DEPS}
              PROFILE_DEPS ${args_PROFILE_DEPS}
              LIGHT_DEPS ${args_LIGHT_DEPS}
              HVY_DEPS ${args_HVY_DEPS}
              )
    _lite_cc_test(${TARGET} SRCS ${args_SRCS} DEPS ${deps} ARGS ${args_ARGS})
    target_compile_options(${TARGET} BEFORE PRIVATE -Wno-ignored-qualifiers)
    file(APPEND ${offline_test_registry_file} "${TARGET}\n")

    # collect targets need to compile for lite
    if (NOT args_EXCLUDE_COMPILE_DEPS)
        add_dependencies(lite_compile_deps ${TARGET})
    endif()
endfunction()

set(arm_kernels CACHE INTERNAL "arm kernels")
set(x86_kernels CACHE INTERNAL "x86 kernels")
set(fpga_kernels CACHE INTERNAL "fpga kernels")
set(npu_kernels CACHE INTERNAL "npu kernels")
set(opencl_kernels CACHE INTERNAL "opencl kernels")
set(host_kernels CACHE INTERNAL "host kernels")

set(kernels_src_list "${CMAKE_BINARY_DIR}/kernels_src_list.txt")
file(WRITE ${kernels_src_list} "") # clean
# add a kernel for some specific device
# device: one of (Host, ARM, X86, NPU, FPGA, OPENCL, CUDA)
# level: one of (basic, extra)
function(add_kernel TARGET device level)
    set(options "")
    set(oneValueArgs "")
    set(multiValueArgs SRCS DEPS X86_DEPS CUDA_DEPS CL_DEPS ARM_DEPS FPGA_DEPS PROFILE_DEPS
        LIGHT_DEPS HVY_DEPS EXCLUDE_COMPILE_DEPS
        ARGS)
    cmake_parse_arguments(args "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if ("${level}" STREQUAL "extra" AND (NOT LITE_BUILD_EXTRA))
        return()
    endif()

    if ("${device}" STREQUAL "Host")
        set(host_kernels "${host_kernels};${TARGET}" CACHE INTERNAL "")
    endif()
    if ("${device}" STREQUAL "ARM")
        if (NOT LITE_WITH_ARM)
            return()
        endif()
        set(arm_kernels "${arm_kernels};${TARGET}" CACHE INTERNAL "")
    endif()
    if ("${device}" STREQUAL "X86")
        if (NOT LITE_WITH_X86)
            return()
        endif()
        set(x86_kernels "${x86_kernels};${TARGET}" CACHE INTERNAL "")
    endif()
    if ("${device}" STREQUAL "NPU")
        if (NOT LITE_WITH_NPU)
            return()
        endif()
        set(npu_kernels "${npu_kernels};${TARGET}" CACHE INTERNAL "")
    endif()
    if ("${device}" STREQUAL "FPGA")
        if (NOT LITE_WITH_FPGA)
            return()
        endif()
        set(fpga_kernels "${fpga_kernels};${TARGET}" CACHE INTERNAL "")
    endif()
    if ("${device}" STREQUAL "OPENCL")
        if (NOT LITE_WITH_OPENCL)
            return()
        endif()
        set(opencl_kernels "${opencl_kernels};${TARGET}" CACHE INTERNAL "")
    endif()

    foreach(src ${args_SRCS})
        file(APPEND ${kernels_src_list} "${CMAKE_CURRENT_SOURCE_DIR}/${src}\n")
    endforeach()

    lite_cc_library(${TARGET} SRCS ${args_SRCS}
              DEPS ${args_DEPS}
              X86_DEPS ${args_X86_DEPS}
              CUDA_DEPS ${args_CUDA_DEPS}
              CL_DEPS ${args_CL_DEPS}
              ARM_DEPS ${args_ARM_DEPS}
              FPGA_DEPS ${args_FPGA_DEPS}
              PROFILE_DEPS ${args_PROFILE_DEPS}
              LIGHT_DEPS ${args_LIGHT_DEPS}
              HVY_DEPS ${args_HVY_DEPS}
      )
endfunction()

set(ops CACHE INTERNAL "ops")
set(ops_src_list "${CMAKE_BINARY_DIR}/ops_src_list.txt")
file(WRITE ${ops_src_list} "") # clean
# add an operator
# level: one of (basic, extra)
function(add_operator TARGET level)
    set(options "")
    set(oneValueArgs "")
    set(multiValueArgs SRCS DEPS X86_DEPS CUDA_DEPS CL_DEPS ARM_DEPS FPGA_DEPS PROFILE_DEPS
        LIGHT_DEPS HVY_DEPS EXCLUDE_COMPILE_DEPS
        ARGS)
    cmake_parse_arguments(args "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if ("${level}" STREQUAL "extra" AND (NOT LITE_BUILD_EXTRA))
        return()
    endif()

    set(ops "${ops};${TARGET}" CACHE INTERNAL "source")

    foreach(src ${args_SRCS})
      file(APPEND ${ops_src_list} "${CMAKE_CURRENT_SOURCE_DIR}/${src}\n")
    endforeach()

    lite_cc_library(${TARGET} SRCS ${args_SRCS}
              DEPS ${args_DEPS}
              X86_DEPS ${args_X86_DEPS}
              CUDA_DEPS ${args_CUDA_DEPS}
              CL_DEPS ${args_CL_DEPS}
              ARM_DEPS ${args_ARM_DEPS}
              FPGA_DEPS ${args_FPGA_DEPS}
              PROFILE_DEPS ${args_PROFILE_DEPS}
              LIGHT_DEPS ${args_LIGHT_DEPS}
              HVY_DEPS ${args_HVY_DEPS}
      )
endfunction()


# Bundle several static libraries into one.
function(bundle_static_library tgt_name bundled_tgt_name fake_target)
  list(APPEND static_libs ${tgt_name})

  function(_recursively_collect_dependencies input_target)
    set(_input_link_libraries LINK_LIBRARIES)
    get_target_property(_input_type ${input_target} TYPE)
    if (${_input_type} STREQUAL "INTERFACE_LIBRARY")
      set(_input_link_libraries INTERFACE_LINK_LIBRARIES)
    endif()
    get_target_property(public_dependencies ${input_target} ${_input_link_libraries})
    foreach(dependency IN LISTS public_dependencies)
      if(TARGET ${dependency})
        get_target_property(alias ${dependency} ALIASED_TARGET)
        if (TARGET ${alias})
          set(dependency ${alias})
        endif()
        get_target_property(_type ${dependency} TYPE)
        if (${_type} STREQUAL "STATIC_LIBRARY")
          list(APPEND static_libs ${dependency})
        endif()

        get_property(library_already_added
          GLOBAL PROPERTY _${tgt_name}_static_bundle_${dependency})
        if (NOT library_already_added)
          set_property(GLOBAL PROPERTY _${tgt_name}_static_bundle_${dependency} ON)
          _recursively_collect_dependencies(${dependency})
        endif()
      endif()
    endforeach()
    set(static_libs ${static_libs} PARENT_SCOPE)
  endfunction()

  _recursively_collect_dependencies(${tgt_name})

  list(REMOVE_DUPLICATES static_libs)

  set(bundled_tgt_full_name
    ${CMAKE_BINARY_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}${bundled_tgt_name}${CMAKE_STATIC_LIBRARY_SUFFIX})

  #message(STATUS "bundled_tgt_full_name: ${bundled_tgt_full_name}")

  if(NOT IOS)
    file(WRITE ${CMAKE_BINARY_DIR}/${bundled_tgt_name}.ar.in
      "CREATE ${bundled_tgt_full_name}\n" )

    foreach(tgt IN LISTS static_libs)
      file(APPEND ${CMAKE_BINARY_DIR}/${bundled_tgt_name}.ar.in
        "ADDLIB $<TARGET_FILE:${tgt}>\n")
    endforeach()

    file(APPEND ${CMAKE_BINARY_DIR}/${bundled_tgt_name}.ar.in "SAVE\n")
    file(APPEND ${CMAKE_BINARY_DIR}/${bundled_tgt_name}.ar.in "END\n")

    file(GENERATE
      OUTPUT ${CMAKE_BINARY_DIR}/${bundled_tgt_name}.ar
      INPUT ${CMAKE_BINARY_DIR}/${bundled_tgt_name}.ar.in)

    set(ar_tool ${CMAKE_AR})
    if (CMAKE_INTERPROCEDURAL_OPTIMIZATION)
      set(ar_tool ${CMAKE_CXX_COMPILER_AR})
    endif()

    add_custom_command(
      COMMAND ${ar_tool} -M < ${CMAKE_BINARY_DIR}/${bundled_tgt_name}.ar
      OUTPUT ${bundled_tgt_full_name}
      COMMENT "Bundling ${bundled_tgt_name}"
      VERBATIM)
  else()
    foreach(lib ${static_libs})
      set(libfiles ${libfiles} $<TARGET_FILE:${lib}>)
    endforeach()
    add_custom_command(
      COMMAND /usr/bin/libtool -static -o ${bundled_tgt_full_name} ${libfiles}
      OUTPUT ${bundled_tgt_full_name}
    )
  endif()

  add_custom_target(${fake_target} ALL DEPENDS ${bundled_tgt_full_name})
  add_dependencies(${fake_target} ${tgt_name})

  add_library(${bundled_tgt_name} STATIC IMPORTED)
  set_target_properties(${bundled_tgt_name}
    PROPERTIES
      IMPORTED_LOCATION ${bundled_tgt_full_name}
      INTERFACE_INCLUDE_DIRECTORIES $<TARGET_PROPERTY:${tgt_name},INTERFACE_INCLUDE_DIRECTORIES>)
  add_dependencies(${bundled_tgt_name} ${fake_target})

endfunction()
