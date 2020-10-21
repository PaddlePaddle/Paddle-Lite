set(LITE_URL "http://paddle-inference-dist.bj.bcebos.com" CACHE STRING "inference download url")

function(lite_download_and_uncompress INSTALL_DIR URL FILENAME)
    message(STATUS "Download inference test stuff: ${FILENAME}")
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
  set(multiValueArgs DEPS X86_DEPS CUDA_DEPS ARM_DEPS PROFILE_DEPS LIGHT_DEPS HVY_DEPS CL_DEPS FPGA_DEPS BM_DEPS RKNPU_DEPS NPU_DEPS XPU_DEPS MLU_DEPS HUAWEI_ASCEND_NPU_DEPS IMAGINATION_NNA_DEPS APU_DEPS CV_DEPS ARGS)
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

  if (LITE_WITH_APU)
    foreach(var ${lite_deps_APU_DEPS})
      set(deps ${deps} ${var})
    endforeach(var)
  endif()

  if (LITE_WITH_RKNPU)
    foreach(var ${lite_deps_RKNPU_DEPS})
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

  if (LITE_WITH_IMAGINATION_NNA)
    foreach(var ${lite_deps_IMAGINATION_NNA_DEPS})
	    set(deps ${deps} ${var})
    endforeach(var)
  endif()

  if (LITE_WITH_HUAWEI_ASCEND_NPU)
    foreach(var ${lite_deps_HUAWEI_ASCEND_NPU_DEPS})
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
#  CV_DEPS:       LITE_WITH_CV
function(lite_cc_library TARGET)
    set(options SHARED shared STATIC static MODULE module)
    set(oneValueArgs "")
    set(multiValueArgs SRCS DEPS X86_DEPS CUDA_DEPS CL_DEPS ARM_DEPS FPGA_DEPS BM_DEPS IMAGINATION_NNA_DEPS RKNPU_DEPS NPU_DEPS XPU_DEPS MLU_DEPS HUAWEI_ASCEND_NPU_DEPS APU_DEPS CV_DEPS PROFILE_DEPS LIGHT_DEPS
      HVY_DEPS EXCLUDE_COMPILE_DEPS ARGS)
    cmake_parse_arguments(args "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    set(deps "")
    lite_deps(deps
            DEPS ${args_DEPS}
            X86_DEPS ${args_X86_DEPS}
            CUDA_DEPS ${args_CUDA_DEPS}
            CL_DEPS ${args_CL_DEPS}
            BM_DEPS ${args_BM_DEPS}
            IMAGINATION_NNA_DEPS ${args_IMAGINATION_NNA_DEPS}
            RKNPU_DEPS ${args_RKNPU_DEPS}
            ARM_DEPS ${args_ARM_DEPS}
            CV_DEPS ${args_CV_DEPS}
            FPGA_DEPS ${args_FPGA_DEPS}
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
    set(multiValueArgs SRCS DEPS X86_DEPS CUDA_DEPS CL_DEPS ARM_DEPS FPGA_DEPS BM_DEPS IMAGINATION_NNA_DEPS RKNPU NPU_DEPS XPU_DEPS MLU_DEPS HUAWEI_ASCEND_NPU_DEPS APU_DEPS PROFILE_DEPS
      LIGHT_DEPS HVY_DEPS EXCLUDE_COMPILE_DEPS CV_DEPS ARGS)
    cmake_parse_arguments(args "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    set(deps "")
    lite_deps(deps
            DEPS ${args_DEPS}
            X86_DEPS ${args_X86_DEPS}
            CUDA_DEPS ${args_CUDA_DEPS}
            CL_DEPS ${args_CL_DEPS}
            ARM_DEPS ${args_ARM_DEPS}
            FPGA_DEPS ${args_FPGA_DEPS}
            NPU_DEPS ${args_NPU_DEPS}
            APU_DEPS ${args_APU_DEPS}
            XPU_DEPS ${args_XPU_DEPS}
            RKNPU_DEPS ${args_RKNPU_DEPS}
            BM_DEPS ${args_BM_DEPS}
            IMAGINATION_NNA_DEPS ${args_IMAGINATION_NNA_DEPS}
            PROFILE_DEPS ${args_PROFILE_DEPS}
            LIGHT_DEPS ${args_LIGHT_DEPS}
            HVY_DEPS ${args_HVY_DEPS}
            CV_DEPS ${CV_DEPS}
            MLU_DEPS ${args_MLU_DEPS}
            HUAWEI_ASCEND_NPU_DEPS ${args_HUAWEI_ASCEND_NPU_DEPS}
            )
    cc_binary(${TARGET} SRCS ${args_SRCS} DEPS ${deps})
    if(NOT WIN32)
      target_compile_options(${TARGET} BEFORE PRIVATE -Wno-ignored-qualifiers)
    endif()
    if (NOT APPLE)
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
    set(multiValueArgs SRCS DEPS X86_DEPS CUDA_DEPS CL_DEPS ARM_DEPS FPGA_DEPS BM_DEPS IMAGINATION_NNA_DEPS RKNPU_DEPS NPU_DEPS XPU_DEPS MLU_DEPS HUAWEI_ASCEND_NPU_DEPS APU_DEPS PROFILE_DEPS
        LIGHT_DEPS HVY_DEPS EXCLUDE_COMPILE_DEPS CV_DEPS
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
              NPU_DEPS ${args_NPU_DEPS}
              APU_DEPS ${args_APU_DEPS}
              XPU_DEPS ${args_XPU_DEPS}
              RKNPU_DEPS ${args_RKNPU_DEPS}
              BM_DEPS ${args_BM_DEPS}
              IMAGINATION_NNA_DEPS ${args_IMAGINATION_NNA_DEPS}
              PROFILE_DEPS ${args_PROFILE_DEPS}
              LIGHT_DEPS ${args_LIGHT_DEPS}
              HVY_DEPS ${args_HVY_DEPS}
              CV_DEPS ${args_CV_DEPS}
              MLU_DEPS ${args_MLU_DEPS}
              HUAWEI_ASCEND_NPU_DEPS ${args_HUAWEI_ASCEND_NPU_DEPS}
              )
    _lite_cc_test(${TARGET} SRCS ${args_SRCS} DEPS ${deps} ARGS ${args_ARGS})
    # strip binary target to reduce size
    if(NOT "${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
        add_custom_command(TARGET ${TARGET} POST_BUILD
                COMMAND "${CMAKE_STRIP}" -s
                "${TARGET}"
                COMMENT "Strip debug symbols done on final executable file.")
    endif()
    if(NOT WIN32)
      target_compile_options(${TARGET} BEFORE PRIVATE -Wno-ignored-qualifiers)
    endif()
    file(APPEND ${offline_test_registry_file} "${TARGET}\n")

    # collect targets need to compile for lite
    if (NOT args_EXCLUDE_COMPILE_DEPS)
        add_dependencies(lite_compile_deps ${TARGET})
    endif()
endfunction()

set(arm_kernels CACHE INTERNAL "arm kernels")
set(x86_kernels CACHE INTERNAL "x86 kernels")
set(cuda_kernels CACHE INTERNAL "cuda kernels")
set(fpga_kernels CACHE INTERNAL "fpga kernels")
set(npu_kernels CACHE INTERNAL "npu kernels")
set(apu_kernels CACHE INTERNAL "apu kernels")
set(xpu_kernels CACHE INTERNAL "xpu kernels")
set(mlu_kernels CACHE INTERNAL "mlu kernels")
set(huawei_ascend_npu_kernels CACHE INTERNAL "huawei_ascend_npu kernels")
set(bm_kernels CACHE INTERNAL "bm kernels")
set(imagination_nna_kernels CACHE INTERNAL "imagination_nna kernels")
set(rknpu_kernels CACHE INTERNAL "rknpu kernels")
set(opencl_kernels CACHE INTERNAL "opencl kernels")
set(host_kernels CACHE INTERNAL "host kernels")

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
# device: one of (Host, ARM, X86, NPU, MLU, HUAWEI_ASCEND_NPU, APU, FPGA, OPENCL, CUDA, BM, RKNPU IMAGINATION_NNA)
# level: one of (basic, extra)
function(add_kernel TARGET device level)
    set(options "")
    set(oneValueArgs "")
    set(multiValueArgs SRCS DEPS X86_DEPS CUDA_DEPS CL_DEPS ARM_DEPS FPGA_DEPS BM_DEPS IMAGINATION_NNA_DEPS RKNPU_DEPS NPU_DEPS XPU_DEPS MLU_DEPS HUAWEI_ASCEND_NPU_DEPS APU_DEPS PROFILE_DEPS
        LIGHT_DEPS HVY_DEPS EXCLUDE_COMPILE_DEPS
        ARGS)
    cmake_parse_arguments(args "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if(LITE_BUILD_TAILOR)
      foreach(src ${args_SRCS})
        list (FIND tailored_kernels_list ${src} _index)
        if (${_index} EQUAL -1)
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


    if ("${device}" STREQUAL "Host")
       if (LITE_ON_MODEL_OPTIMIZE_TOOL)
            foreach(src ${args_SRCS})
                file(APPEND ${fake_kernels_src_list} "${CMAKE_CURRENT_SOURCE_DIR}/${src}\n")
            endforeach()
            return()
        endif()
        set(host_kernels "${host_kernels};${TARGET}" CACHE INTERNAL "")
    endif()
    if ("${device}" STREQUAL "ARM")
        if (NOT LITE_WITH_ARM)
            foreach(src ${args_SRCS})
                file(APPEND ${fake_kernels_src_list} "${CMAKE_CURRENT_SOURCE_DIR}/${src}\n")
            endforeach()
            return()
        endif()
        set(arm_kernels "${arm_kernels};${TARGET}" CACHE INTERNAL "")
    endif()
    if ("${device}" STREQUAL "X86")
        if (NOT LITE_WITH_X86 OR LITE_ON_MODEL_OPTIMIZE_TOOL)
            foreach(src ${args_SRCS})
                file(APPEND ${fake_kernels_src_list} "${CMAKE_CURRENT_SOURCE_DIR}/${src}\n")
            endforeach()
            return()
        endif()
        set(x86_kernels "${x86_kernels};${TARGET}" CACHE INTERNAL "")
    endif()
    if ("${device}" STREQUAL "NPU")
        if (NOT LITE_WITH_NPU)
            foreach(src ${args_SRCS})
                file(APPEND ${fake_kernels_src_list} "${CMAKE_CURRENT_SOURCE_DIR}/${src}\n")
            endforeach()
            return()
        endif()
        set(npu_kernels "${npu_kernels};${TARGET}" CACHE INTERNAL "")
    endif()
    if ("${device}" STREQUAL "APU")
        if (NOT LITE_WITH_APU)
            foreach(src ${args_SRCS})
                file(APPEND ${fake_kernels_src_list} "${CMAKE_CURRENT_SOURCE_DIR}/${src}\n")
            endforeach()
            return()
        endif()
        set(apu_kernels "${apu_kernels};${TARGET}" CACHE INTERNAL "")
    endif()
    if ("${device}" STREQUAL "XPU")
        if (NOT LITE_WITH_XPU)
            foreach(src ${args_SRCS})
                file(APPEND ${fake_kernels_src_list} "${CMAKE_CURRENT_SOURCE_DIR}/${src}\n")
            endforeach()
            return()
        endif()
        set(xpu_kernels "${xpu_kernels};${TARGET}" CACHE INTERNAL "")
    endif()
    if ("${device}" STREQUAL "FPGA")
        if (NOT LITE_WITH_FPGA)
            foreach(src ${args_SRCS})
                file(APPEND ${fake_kernels_src_list} "${CMAKE_CURRENT_SOURCE_DIR}/${src}\n")
            endforeach()
            return()
        endif()
        set(fpga_kernels "${fpga_kernels};${TARGET}" CACHE INTERNAL "")
    endif()
    if ("${device}" STREQUAL "BM")
        if (NOT LITE_WITH_BM)
            foreach(src ${args_SRCS})
                file(APPEND ${fake_kernels_src_list} "${CMAKE_CURRENT_SOURCE_DIR}/${src}\n")
            endforeach()
            return()
        endif()
        set(bm_kernels "${bm_kernels};${TARGET}" CACHE INTERNAL "")
    endif()
    if ("${device}" STREQUAL "RKNPU")
        if (NOT LITE_WITH_RKNPU)
            foreach(src ${args_SRCS})
                file(APPEND ${fake_kernels_src_list} "${CMAKE_CURRENT_SOURCE_DIR}/${src}\n")
            endforeach()
            return()
        endif()
        set(rknpu_kernels "${rknpu_kernels};${TARGET}" CACHE INTERNAL "")
    endif()
    if ("${device}" STREQUAL "MLU")
        if (NOT LITE_WITH_MLU)
            foreach(src ${args_SRCS})
                file(APPEND ${fake_kernels_src_list} "${CMAKE_CURRENT_SOURCE_DIR}/${src}\n")
            endforeach()
            return()
        endif()
        set(mlu_kernels "${mlu_kernels};${TARGET}" CACHE INTERNAL "")
    endif()
    if ("${device}" STREQUAL "IMAGINATION_NNA")
        if (NOT LITE_WITH_IMAGINATION_NNA)
            foreach(src ${args_SRCS})
                file(APPEND ${fake_kernels_src_list} "${CMAKE_CURRENT_SOURCE_DIR}/${src}\n")
            endforeach()
            return()
        endif()
        set(imagination_nna_kernels "${imagination_nna_kernels};${TARGET}" CACHE INTERNAL "")
    endif()

    if ("${device}" STREQUAL "HUAWEI_ASCEND_NPU")
        if (NOT LITE_WITH_HUAWEI_ASCEND_NPU)
            foreach(src ${args_SRCS})
                file(APPEND ${fake_kernels_src_list} "${CMAKE_CURRENT_SOURCE_DIR}/${src}\n")
            endforeach()
            return()
        endif()
        set(huawei_ascend_npu_kernels "${huawei_ascend_npu_kernels};${TARGET}" CACHE INTERNAL "")
    endif()
    if ("${device}" STREQUAL "OPENCL")
        if (NOT LITE_WITH_OPENCL)
            foreach(src ${args_SRCS})
                file(APPEND ${fake_kernels_src_list} "${CMAKE_CURRENT_SOURCE_DIR}/${src}\n")
            endforeach()
            return()
        endif()
        set(opencl_kernels "${opencl_kernels};${TARGET}" CACHE INTERNAL "")
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
        nv_library(${TARGET} SRCS ${args_SRCS} DEPS ${args_DEPS})
        return()
    endif()

    # the source list will collect for paddle_use_kernel.h code generation.
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
              NPU_DEPS ${args_NPU_DEPS}
              APU_DEPS ${args_APU_DEPS}
              XPU_DEPS ${args_XPU_DEPS}
              RKNPU_DEPS ${args_RKNPU_DEPS}
              BM_DEPS ${args_BM_DEPS}
              MLU_DEPS ${args_MLU_DEPS}
              IMAGINATION_NNA_DEPS ${args_IMAGINATION_NNA_DEPS}
              HUAWEI_ASCEND_NPU_DEPS ${args_HUAWEI_ASCEND_NPU_DEPS}
              PROFILE_DEPS ${args_PROFILE_DEPS}
              LIGHT_DEPS ${args_LIGHT_DEPS}
              HVY_DEPS ${args_HVY_DEPS}
      )
endfunction()

set(ops CACHE INTERNAL "ops")
set(ops_src_list "${CMAKE_BINARY_DIR}/ops_src_list.txt")
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
    set(multiValueArgs SRCS DEPS X86_DEPS CUDA_DEPS CL_DEPS ARM_DEPS FPGA_DEPS BM_DEPS IMAGINATION_NNA_DEPS NPU_DEPS XPU_DEPS MLU_DEPS HUAWEI_ASCEND_NPU_DEPS APU_DEPS PROFILE_DEPS
        LIGHT_DEPS HVY_DEPS EXCLUDE_COMPILE_DEPS
        ARGS)
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
    endforeach()

    set(ops "${ops};${TARGET}" CACHE INTERNAL "source")

    lite_cc_library(${TARGET} SRCS ${args_SRCS}
              DEPS ${args_DEPS}
              X86_DEPS ${args_X86_DEPS}
              CUDA_DEPS ${args_CUDA_DEPS}
              CL_DEPS ${args_CL_DEPS}
              ARM_DEPS ${args_ARM_DEPS}
              FPGA_DEPS ${args_FPGA_DEPS}
              NPU_DEPS ${args_NPU_DEPS}
              APU_DEPS ${args_APU_DEPS}
              XPU_DEPS ${args_XPU_DEPS}
              RKNPU_DEPS ${args_RKNPU_DEPS}
              BM_DEPS ${args_BM_DEPS}
              MLU_DEPS ${args_MLU_DEPS}
              IMAGINATION_NNA_DEPS ${args_IMAGINATION_NNA_DEPS}
              HUAWEI_ASCEND_NPU_DEPS ${args_HUAWEI_ASCEND_NPU_DEPS}
              PROFILE_DEPS ${args_PROFILE_DEPS}
              LIGHT_DEPS ${args_LIGHT_DEPS}
              HVY_DEPS ${args_HVY_DEPS}
      )
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
    ${CMAKE_BINARY_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}${bundled_tgt_name}${CMAKE_STATIC_LIBRARY_SUFFIX})

  message(STATUS "bundled_tgt_full_name:  ${CMAKE_BINARY_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}${bundled_tgt_name}${CMAKE_STATIC_LIBRARY_SUFFIX}")
  
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
