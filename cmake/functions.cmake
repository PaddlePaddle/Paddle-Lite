# add a kernel for some specific device
# device: one of (Host, ARM, X86, NPU, MLU, HUAWEI_ASCEND_NPU, APU, FPGA, OPENCL, CUDA, BM, RKNPU IMAGINATION_NNA)
# level: one of (basic, extra)
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
    if ("${device}" STREQUAL "INTEL_FPGA")
        if (NOT LITE_WITH_INTEL_FPGA)
            foreach(src ${args_SRCS})
                file(APPEND ${fake_kernels_src_list} "${CMAKE_CURRENT_SOURCE_DIR}/${src}\n")
            endforeach()
            return()
        endif()
        set(intel_fpga_kernels "${intel_fpga_kernels};${TARGET}" CACHE INTERNAL "")
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
    if ("${device}" STREQUAL "NNADAPTER")
        if (NOT LITE_WITH_NNADAPTER)
            foreach(src ${args_SRCS})
                file(APPEND ${fake_kernels_src_list} "${CMAKE_CURRENT_SOURCE_DIR}/${src}\n")
            endforeach()
            return()
        endif()
        set(nnadapter_kernels "${nnadapter_kernels};${TARGET}" CACHE INTERNAL "")
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

    if ("${device}" STREQUAL "METAL")
        if (NOT LITE_WITH_METAL)
            foreach(src ${args_SRCS})
                file(APPEND ${fake_kernels_src_list} "${CMAKE_CURRENT_SOURCE_DIR}/${src}\n")
            endforeach()
            return()
        endif()
        set(metal_kernels "${metal_kernels};${TARGET}" CACHE INTERNAL "")
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

    if (NOT LITE_BUILD_TAILOR)
      lite_cc_library(${TARGET} SRCS ${args_SRCS}
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
              MLU_DEPS ${args_MLU_DEPS}
              IMAGINATION_NNA_DEPS ${args_IMAGINATION_NNA_DEPS}
              HUAWEI_ASCEND_NPU_DEPS ${args_HUAWEI_ASCEND_NPU_DEPS}
              PROFILE_DEPS ${args_PROFILE_DEPS}
              LIGHT_DEPS ${args_LIGHT_DEPS}
              HVY_DEPS ${args_HVY_DEPS})
    else()
      lite_cc_library(${TARGET} SRCS ${dst_file}
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
              MLU_DEPS ${args_MLU_DEPS}
              IMAGINATION_NNA_DEPS ${args_IMAGINATION_NNA_DEPS}
              NNADAPTER_DEPS ${args_NNADAPTER_DEPS}
              HUAWEI_ASCEND_NPU_DEPS ${args_HUAWEI_ASCEND_NPU_DEPS}
              PROFILE_DEPS ${args_PROFILE_DEPS}
              LIGHT_DEPS ${args_LIGHT_DEPS}
              HVY_DEPS ${args_HVY_DEPS}
      )
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

