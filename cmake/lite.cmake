set(LITE_URL "http://paddle-inference-dist.bj.bcebos.com" CACHE STRING "inference download url")

function(lite_download_and_uncompress INSTALL_DIR URL FILENAME)
  set(options "")
  set(oneValueArgs MODEL_PATH)
  set(multiValueArgs "")
  cmake_parse_arguments(args "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  if(DEFINED args_MODEL_PATH)
    set(FILE_PATH ${args_MODEL_PATH}/${FILENAME})
    set(PREFIX ${INSTALL_DIR}/${args_MODEL_PATH})
    set(DOWNLOAD_DIR ${INSTALL_DIR}/${args_MODEL_PATH})
  else()
    set(FILE_PATH ${FILENAME})
    set(PREFIX ${INSTALL_DIR})
    set(DOWNLOAD_DIR ${INSTALL_DIR})
  endif()

  message(STATUS "Download inference test stuff: ${FILE_PATH}")
  string(REGEX REPLACE "[-%./]" "_" FILENAME_EX ${FILE_PATH})
  set(EXTERNAL_PROJECT_NAME "extern_lite_download_${FILENAME_EX}")
  set(UNPACK_DIR "${INSTALL_DIR}/src/${EXTERNAL_PROJECT_NAME}")
  ExternalProject_Add(
            ${EXTERNAL_PROJECT_NAME}
            ${EXTERNAL_PROJECT_LOG_ARGS}
            PREFIX                ${PREFIX}
            DOWNLOAD_COMMAND      wget --no-check-certificate -q -O ${INSTALL_DIR}/${FILE_PATH} ${URL}/${FILE_PATH} && ${CMAKE_COMMAND} -E tar xzf ${INSTALL_DIR}/${FILE_PATH} && rm -f ${INSTALL_DIR}/${FILE_PATH}
            DOWNLOAD_DIR          ${DOWNLOAD_DIR}
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
  set(multiValueArgs DEPS X86_DEPS CUDA_DEPS ARM_DEPS PROFILE_DEPS LIGHT_DEPS HVY_DEPS CL_DEPS METAL_DEPS FPGA_DEPS INTEL_FPGA_DEPS BM_DEPS RKNPU_DEPS NPU_DEPS XPU_DEPS MLU_DEPS HUAWEI_ASCEND_NPU_DEPS IMAGINATION_NNA_DEPS APU_DEPS NNADAPTER_DEPS CV_DEPS ARGS)
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
    if(LITE_WITH_CV)
      foreach(var ${lite_deps_CV_DEPS})
        set(deps ${deps} ${var})
      endforeach(var)
    endif()
  endif()

  if(LITE_WITH_PROFILE)
    foreach(var ${lite_deps_PROFILE_DEPS})
      set(deps ${deps} ${var})
    endforeach(var)
  endif()

  if(LITE_WITH_ARM)
    foreach(var ${lite_deps_LIGHT_DEPS})
      set(deps ${deps} ${var})
    endforeach(var)
  endif()

  if (NOT LITE_WITH_ARM)
    foreach(var ${lite_deps_HVY_DEPS})
      set(deps ${deps} ${var})
    endforeach(var)
  endif()

  if (LITE_WITH_OPENCL)
    foreach(var ${lite_deps_CL_DEPS})
      set(deps ${deps} ${var})
    endforeach(var)
  endif()

  if (LITE_WITH_METAL)
    foreach(var ${lite_deps_METAL_DEPS})
      set(deps ${deps} ${var})
    endforeach(var)
  endif()

  if (LITE_WITH_FPGA)
    foreach(var ${lite_deps_FPGA_DEPS})
      set(deps ${deps} ${var})
    endforeach(var)
  endif()

  if (LITE_WITH_INTEL_FPGA)
    foreach(var ${lite_deps_INTEL_FPGA_DEPS})
      set(deps ${deps} ${var})
    endforeach(var)
  endif()

  if (LITE_WITH_NPU)
    foreach(var ${lite_deps_NPU_DEPS})
      set(deps ${deps} ${var})
    endforeach(var)
  endif()

  if (LITE_WITH_XPU)
    foreach(var ${lite_deps_XPU_DEPS})
      set(deps ${deps} ${var})
    endforeach(var)
  endif()

  if (LITE_WITH_BM)
    foreach(var ${lite_deps_BM_DEPS})
      set(deps ${deps} ${var})
    endforeach(var)
  endif()

  if (LITE_WITH_MLU)
    foreach(var ${lite_deps_MLU_DEPS})
      set(deps ${deps} ${var})
    endforeach(var)
  endif()

  if (LITE_WITH_NNADAPTER)
    foreach(var ${lite_deps_NNADAPTER_DEPS})
      set(deps ${deps} ${var})
    endforeach(var)
  endif()

  set(${TARGET} ${deps} PARENT_SCOPE)
endfunction()


# A fake target to include all the libraries and tests the lite module depends.
add_custom_target(lite_compile_deps COMMAND echo 1)

# Add names for lite libraries for latter compile. We use this name list to avoid compiling
# the whole fluid project to accelerate the compile speed.
set(offline_lib_registry_file "${PADDLE_BINARY_DIR}/lite_libs.txt")
file(WRITE ${offline_lib_registry_file} "") # clean

# cc_library with branch support.
# The branches:
#  X86_DEPS: works only when LITE_WITH_X86 is ON.
#  CUDA_DEPS:     LITE_WITH_CUDA
#  ARM_DEPS:      LITE_WITH_ARM
#  PROFILE_DEPS:  LITE_WITH_PROFILE
#  EXCLUDE_COMPILE_DEPS: TARGET will not be included in lite_compile_deps if this is not None
#  CV_DEPS:       LITE_WITH_CV
function(lite_cc_library TARGET)
    set(options SHARED shared STATIC static MODULE module)
    set(oneValueArgs "")
    set(multiValueArgs SRCS DEPS X86_DEPS CUDA_DEPS CL_DEPS METAL_DEPS ARM_DEPS FPGA_DEPS INTEL_FPGA_DEPS BM_DEPS IMAGINATION_NNA_DEPS RKNPU_DEPS NPU_DEPS XPU_DEPS MLU_DEPS HUAWEI_ASCEND_NPU_DEPS APU_DEPS NNADAPTER_DEPS CV_DEPS PROFILE_DEPS LIGHT_DEPS
      HVY_DEPS EXCLUDE_COMPILE_DEPS ARGS)
    cmake_parse_arguments(args "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    set(deps "")
    lite_deps(deps
            DEPS ${args_DEPS}
            X86_DEPS ${args_X86_DEPS}
            CUDA_DEPS ${args_CUDA_DEPS}
            CL_DEPS ${args_CL_DEPS}
            METAL_DEPS ${args_METAL_DEPS}
            BM_DEPS ${args_BM_DEPS}
            IMAGINATION_NNA_DEPS ${args_IMAGINATION_NNA_DEPS}
            NNADAPTER_DEPS ${args_NNADAPTER_DEPS}
            RKNPU_DEPS ${args_RKNPU_DEPS}
            ARM_DEPS ${args_ARM_DEPS}
            CV_DEPS ${args_CV_DEPS}
            FPGA_DEPS ${args_FPGA_DEPS}
            INTEL_FPGA_DEPS ${args_INTEL_FPGA_DEPS}
            NPU_DEPS ${args_NPU_DEPS}
            APU_DEPS ${args_APU_DEPS}
            XPU_DEPS ${args_XPU_DEPS}
            PROFILE_DEPS ${args_PROFILE_DEPS}
            LIGHT_DEPS ${args_LIGHT_DEPS}
            HVY_DEPS ${args_HVY_DEPS}
            MLU_DEPS ${args_MLU_DEPS}
            HUAWEI_ASCEND_NPU_DEPS ${args_HUAWEI_ASCEND_NPU_DEPS}
            )

    if (args_SHARED OR ARGS_shared)
        cc_library(${TARGET} SRCS ${args_SRCS} DEPS ${deps} SHARED)
    elseif (args_MODULE OR ARGS_module)
        add_library(${TARGET} MODULE ${args_SRCS})
        add_dependencies(${TARGET} ${deps} ${args_DEPS})
    else()
        cc_library(${TARGET} SRCS ${args_SRCS} DEPS ${deps})
    endif()

    if(NOT WIN32)
      target_compile_options(${TARGET} BEFORE PRIVATE -Wno-ignored-qualifiers)
    endif()
    # collect targets need to compile for lite
    if (args_SRCS AND NOT args_EXCLUDE_COMPILE_DEPS)
        add_dependencies(lite_compile_deps ${TARGET})
    endif()

    # register a library name.
    file(APPEND ${offline_lib_registry_file} "${TARGET}\n")
endfunction()

function(lite_cc_binary TARGET)
    if ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
        set(options " -g ")
    endif()
    set(oneValueArgs "")
    set(multiValueArgs SRCS DEPS X86_DEPS CUDA_DEPS CL_DEPS METAL_DEPS ARM_DEPS FPGA_DEPS INTEL_FPGA_DEPS BM_DEPS IMAGINATION_NNA_DEPS RKNPU NPU_DEPS XPU_DEPS MLU_DEPS HUAWEI_ASCEND_NPU_DEPS APU_DEPS NNADAPTER_DEPS PROFILE_DEPS
      LIGHT_DEPS HVY_DEPS EXCLUDE_COMPILE_DEPS CV_DEPS ARGS)
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
            CV_DEPS ${CV_DEPS}
            MLU_DEPS ${args_MLU_DEPS}
            HUAWEI_ASCEND_NPU_DEPS ${args_HUAWEI_ASCEND_NPU_DEPS}
            )
    cc_binary(${TARGET} SRCS ${args_SRCS} DEPS ${deps})

    # link to paddle-lite static lib automatically
    add_dependencies(${TARGET} bundle_full_api)

    if(NOT WIN32)
      target_link_libraries(${TARGET} ${PADDLE_BINARY_DIR}/libpaddle_api_full_bundled.a)
      target_compile_options(${TARGET} BEFORE PRIVATE -Wno-ignored-qualifiers)
      # openmp dynamic lib is required for mkl
      if(WITH_STATIC_MKL)
        target_link_libraries(${TARGET} -L${MKLML_LIB_DIR} -liomp5)
      endif()
    else()
      target_link_libraries(${TARGET} ${PADDLE_BINARY_DIR}/lite/api/${CMAKE_BUILD_TYPE}/libpaddle_api_full_bundled.lib)
    endif()

    # link to dynamic runtime lib
    if(LITE_WITH_METAL)
        target_link_libraries(${TARGET} ${METAL_LIBRARY} ${GRAPHIC} ${MPS_LIBRARY} ${FOUNDATION_LIBRARY})
    endif()
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
    if(LITE_WITH_INTEL_FPGA)
        target_link_libraries(${TARGET} ${intel_fpga_deps})
    endif()

    if (NOT APPLE AND NOT WIN32 AND NOT EMSCRIPTEN)
        # strip binary target to reduce size
        if(NOT "${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
            add_custom_command(TARGET ${TARGET} POST_BUILD
                    COMMAND "${CMAKE_STRIP}" -s
                    "${TARGET}"
                    COMMENT "Strip debug symbols done on final executable file.")
        endif()
    endif()
    # collect targets need to compile for lite
    if (NOT args_EXCLUDE_COMPILE_DEPS)
        add_dependencies(lite_compile_deps ${TARGET})
    endif()
endfunction()

#only for windows
function(create_static_lib TARGET_NAME)
  set(libs ${ARGN})
  list(REMOVE_DUPLICATES libs)
    set(dummy_index 1)
    set(dummy_offset 1)
    # the dummy target would be consisted of limit size libraries
    set(dummy_limit 60)
    list(LENGTH libs libs_len)

    foreach(lib ${libs})
      list(APPEND dummy_list ${lib})
      list(LENGTH dummy_list listlen)
      if ((${listlen} GREATER ${dummy_limit}) OR (${dummy_offset} EQUAL ${libs_len}))
        merge_static_libs(${TARGET_NAME}_dummy_${dummy_index} ${dummy_list})
        set(dummy_list)
        list(APPEND ${TARGET_NAME}_dummy_list ${TARGET_NAME}_dummy_${dummy_index})
        MATH(EXPR dummy_index "${dummy_index}+1")
      endif()
      MATH(EXPR dummy_offset "${dummy_offset}+1")
    endforeach()
    merge_static_libs(${TARGET_NAME} ${${TARGET_NAME}_dummy_list})
endfunction()

# Bundle several static libraries into one.
function(bundle_static_library tgt_name bundled_tgt_name fake_target)
  list(APPEND static_libs ${tgt_name})
# for x86
  add_dependencies(lite_compile_deps ${fake_target})

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
    ${PADDLE_BINARY_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}${bundled_tgt_name}${CMAKE_STATIC_LIBRARY_SUFFIX})

  message(STATUS "bundled_tgt_full_name:  ${PADDLE_BINARY_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}${bundled_tgt_name}${CMAKE_STATIC_LIBRARY_SUFFIX}")

  if(WIN32)
    set(dummy_tgt_name dummy_${bundled_tgt_name})
    create_static_lib(${bundled_tgt_name} ${static_libs})
    add_custom_target(${fake_target} ALL DEPENDS ${bundled_tgt_name})
    add_dependencies(${fake_target} ${tgt_name})

    add_library(${dummy_tgt_name} STATIC IMPORTED)
    set_target_properties(${dummy_tgt_name}
      PROPERTIES
        IMPORTED_LOCATION ${bundled_tgt_full_name}
        INTERFACE_INCLUDE_DIRECTORIES $<TARGET_PROPERTY:${tgt_name},INTERFACE_INCLUDE_DIRECTORIES>)
    add_dependencies(${dummy_tgt_name} ${fake_target})
    return()
  endif()

  add_custom_target(${fake_target})
  add_dependencies(${fake_target} ${tgt_name})

  if(NOT IOS AND NOT APPLE)
    file(WRITE ${PADDLE_BINARY_DIR}/${bundled_tgt_name}.ar.in
      "CREATE ${bundled_tgt_full_name}\n" )

    foreach(tgt IN LISTS static_libs)
      file(APPEND ${PADDLE_BINARY_DIR}/${bundled_tgt_name}.ar.in
        "ADDLIB $<TARGET_FILE:${tgt}>\n")
    endforeach()

    file(APPEND ${PADDLE_BINARY_DIR}/${bundled_tgt_name}.ar.in "SAVE\n")
    file(APPEND ${PADDLE_BINARY_DIR}/${bundled_tgt_name}.ar.in "END\n")

    file(GENERATE
      OUTPUT ${PADDLE_BINARY_DIR}/${bundled_tgt_name}.ar
      INPUT ${PADDLE_BINARY_DIR}/${bundled_tgt_name}.ar.in)

    set(ar_tool ${CMAKE_AR})
    if (CMAKE_INTERPROCEDURAL_OPTIMIZATION)
      set(ar_tool ${CMAKE_CXX_COMPILER_AR})
    endif()

    add_custom_command(
      TARGET ${fake_target} PRE_BUILD
      COMMAND rm -f ${bundled_tgt_full_name}
      COMMAND ${ar_tool} -M < ${PADDLE_BINARY_DIR}/${bundled_tgt_name}.ar
      COMMENT "Bundling ${bundled_tgt_name}"
      DEPENDS ${tgt_name}
      VERBATIM)
  else()
    foreach(lib ${static_libs})
      set(libfiles ${libfiles} $<TARGET_FILE:${lib}>)
    endforeach()
    add_custom_command(
      TARGET ${fake_target} PRE_BUILD
      COMMAND rm -f ${bundled_tgt_full_name}
      COMMAND /usr/bin/libtool -static -o ${bundled_tgt_full_name} ${libfiles}
      DEPENDS ${tgt_name}
    )
  endif()

  add_library(${bundled_tgt_name} STATIC IMPORTED)
  set_target_properties(${bundled_tgt_name}
    PROPERTIES
      IMPORTED_LOCATION ${bundled_tgt_full_name}
      INTERFACE_INCLUDE_DIRECTORIES $<TARGET_PROPERTY:${tgt_name},INTERFACE_INCLUDE_DIRECTORIES>)
  add_dependencies(${bundled_tgt_name} ${fake_target})

endfunction()
