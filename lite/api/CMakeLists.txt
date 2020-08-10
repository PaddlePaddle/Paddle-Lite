if(LITE_WITH_LIGHT_WEIGHT_FRAMEWORK OR (NOT LITE_WITH_LOG))
  lite_cc_library(place SRCS paddle_place.cc DEPS logging)
else()
  lite_cc_library(place SRCS paddle_place.cc DEPS glog)
endif()

if (LITE_ON_TINY_PUBLISH)
    set(CMAKE_CXX_FLAGS_RELEASE "-Os -DNDEBUG")
    set(CMAKE_C_FLAGS_RELEASE "-Os -DNDEBUG")
endif()

set(light_lib_DEPS light_api paddle_api paddle_api_light)

if ((NOT LITE_ON_TINY_PUBLISH) AND (LITE_WITH_CUDA OR LITE_WITH_X86 OR LITE_WITH_BM OR LITE_WITH_HUAWEI_ASCEND_NPU OR ARM_TARGET_OS STREQUAL "android" OR ARM_TARGET_OS STREQUAL "armlinux"))
    #full api dynamic library
    lite_cc_library(paddle_full_api_shared SHARED SRCS paddle_api.cc light_api.cc cxx_api.cc cxx_api_impl.cc light_api_impl.cc
                  DEPS paddle_api paddle_api_light  paddle_api_full)
    target_sources(paddle_full_api_shared PUBLIC ${__lite_cc_files})
    add_dependencies(paddle_full_api_shared op_list_h kernel_list_h framework_proto op_registry fbs_headers)
    target_link_libraries(paddle_full_api_shared framework_proto op_registry)
    if(LITE_WITH_X86)
        add_dependencies(paddle_full_api_shared xxhash)
        target_link_libraries(paddle_full_api_shared xxhash)
        if (NOT LITE_ON_MODEL_OPTIMIZE_TOOL)
            add_dependencies(paddle_full_api_shared dynload_mklml)
        endif()
        if(WIN32)
             target_link_libraries(paddle_full_api_shared shlwapi.lib)
        endif()
    endif()
    if(LITE_WITH_CUDA)
        target_link_libraries(paddle_full_api_shared ${math_cuda} "-Wl,--whole-archive" ${cuda_kernels} "-Wl,--no-whole-archive")
    endif(LITE_WITH_CUDA)

    #light api dynamic library
    lite_cc_library(paddle_light_api_shared SHARED SRCS paddle_api.cc light_api.cc light_api_impl.cc
                  DEPS ${light_lib_DEPS}
                  ARM_DEPS ${arm_kernels}
                  CV_DEPS paddle_cv_arm
                  NPU_DEPS ${npu_kernels}
                  APU_DEPS ${apu_kernels}
                  RKNPU_DEPS ${rknpu_kernels}
                  HUAWEI_ASCEND_NPU_DEPS ${huawei_ascend_npu_kernels}
                  )

    add_dependencies(paddle_light_api_shared op_list_h kernel_list_h)
    if(WIN32)
        target_link_libraries(paddle_light_api_shared shlwapi.lib)
    endif()
    target_link_libraries(paddle_light_api_shared ${light_lib_DEPS} ${arm_kernels} ${npu_kernels} ${huawei_ascend_npu_kernels} ${rknpu_kernels} ${apu_kernels})
   if(APPLE)
        set(LINK_MAP_FILE "${PADDLE_SOURCE_DIR}/lite/core/exported_symbols.lds")
        set(LINK_FLAGS "-Wl,-exported_symbols_list, ${LINK_MAP_FILE}")
        add_custom_command(OUTPUT ${LINK_MAP_FILE} COMMAND ...)
        add_custom_target(custom_linker_map DEPENDS ${LINK_MAP_FILE})
        set_target_properties(paddle_full_api_shared PROPERTIES LINK_FLAGS ${LINK_FLAGS})
        add_dependencies(paddle_full_api_shared custom_linker_map)
   elseif(NOT WIN32)
        set(LINK_MAP_FILE "${PADDLE_SOURCE_DIR}/lite/core/lite.map")
        set(LINK_FLAGS "-Wl,--version-script ${LINK_MAP_FILE}")
        add_custom_command(OUTPUT ${LINK_MAP_FILE} COMMAND ...)
        add_custom_target(custom_linker_map DEPENDS ${LINK_MAP_FILE})
        set_target_properties(paddle_full_api_shared PROPERTIES LINK_FLAGS ${LINK_FLAGS})
        add_dependencies(paddle_full_api_shared custom_linker_map)
   endif()
else()
    if ((ARM_TARGET_OS STREQUAL "android") OR (ARM_TARGET_OS STREQUAL "armlinux"))
        add_library(paddle_light_api_shared SHARED "")
        target_sources(paddle_light_api_shared PUBLIC ${__lite_cc_files} paddle_api.cc light_api.cc light_api_impl.cc)
        set(TARGET_COMIPILE_FLAGS "-fdata-sections")
        if (NOT (ARM_TARGET_LANG STREQUAL "clang")) #gcc
            set(TARGET_COMIPILE_FLAGS "${TARGET_COMIPILE_FLAGS} -flto")
        endif()
        set_target_properties(paddle_light_api_shared PROPERTIES COMPILE_FLAGS "${TARGET_COMIPILE_FLAGS}")
        add_dependencies(paddle_light_api_shared op_list_h kernel_list_h fbs_headers)
        if (LITE_WITH_NPU)
            # Need to add HIAI runtime libs (libhiai.so) dependency
            target_link_libraries(paddle_light_api_shared ${npu_builder_libs} ${npu_runtime_libs})
        endif()
        if (LITE_WITH_RKNPU)
            # Need to add RKNPU runtime libs dependency
            target_link_libraries(paddle_light_api_shared ${rknpu_builder_libs} ${rknpu_runtime_libs})
        endif()

    endif()
endif()

if (WITH_TESTING)
    lite_cc_library(lite_api_test_helper SRCS lite_api_test_helper.cc
      DEPS scope optimizer target_wrapper_host model_parser program
           ${ops} ${host_kernels}
      CUDA_DEPS ${cuda_kernels}
      X86_DEPS ${x86_kernels}
      XPU_DEPS ${xpu_kernels}
      RKNPU_DEPS ${rknpu_kernels}
      BM_DEPS ${bm_kernels}
      MLU_DEPS ${mlu_kernels}
      HUAWEI_ASCEND_NPU_DEPS ${huawei_ascend_npu_kernels}
      APU_DEPS ${apu_kernels})

endif()
if(LITE_WITH_FPGA)
    set(light_api_deps ${light_api_deps} ${fpga_deps})
    set(cxx_api_deps ${cxx_api_deps} ${fpga_deps})
endif()

if(LITE_WITH_BM)
    set(light_api_deps ${light_api_deps} ${bm_deps})
    set(cxx_api_deps ${cxx_api_deps} ${bm_deps})
endif()

if(LITE_WITH_RKNPU)
    set(light_api_deps ${light_api_deps} ${rknpu_deps})
    set(cxx_api_deps ${cxx_api_deps} ${rknpu_deps})
endif()

if(LITE_WITH_HUAWEI_ASCEND_NPU)
    set(light_api_deps ${light_api_deps} ${huawei_ascend_npu_deps})
    set(cxx_api_deps ${cxx_api_deps} ${huawei_ascend_npu_deps})
endif()

message(STATUS "get ops ${ops}")
message(STATUS "get X86 kernels ${x86_kernels}")
message(STATUS "get CUDA kernels ${cuda_kernels}")
message(STATUS "get Host kernels ${host_kernels}")
message(STATUS "get ARM kernels ${arm_kernels}")
message(STATUS "get OpenCL kernels ${opencl_kernels}")
message(STATUS "get NPU kernels ${npu_kernels}")
message(STATUS "get APU kernels ${apu_kernels}")
message(STATUS "get XPU kernels ${xpu_kernels}")
message(STATUS "get RKNPU kernels ${rknpu_kernels}")
message(STATUS "get FPGA kernels ${fpga_kernels}")
message(STATUS "get BM kernels ${bm_kernels}")
message(STATUS "get MLU kernels ${mlu_kernels}")
message(STATUS "get HUAWEI_ASCEND_NPU kernels ${huawei_ascend_npu_kernels}")

# for full api
if (NOT LITE_ON_TINY_PUBLISH)
    set(cxx_api_deps
    scope optimizer target_wrapper_host model_parser program)
    lite_cc_library(cxx_api
                        SRCS cxx_api.cc
                        DEPS ${cxx_api_deps} ${ops} ${host_kernels} program
                        X86_DEPS ${x86_kernels}
                        CUDA_DEPS ${cuda_kernels}
                        ARM_DEPS ${arm_kernels}
                        CV_DEPS paddle_cv_arm
                        NPU_DEPS ${npu_kernels}
                        XPU_DEPS ${xpu_kernels}
                        APU_DEPS ${apu_kernels}
                        RKNPU_DEPS ${rknpu_kernels}
                        BM_DEPS ${bm_kernels}
                        CL_DEPS ${opencl_kernels}
                        FPGA_DEPS ${fpga_kernels}
                        HUAWEI_ASCEND_NPU_DEPS ${huawei_ascend_npu_kernels})
endif()

# for light api
set(light_api_deps
    scope target_wrapper_host model_parser program)
if(LITE_WITH_CUDA)
    get_property(cuda_deps GLOBAL PROPERTY CUDA_MODULES)
    set(light_api_deps ${light_api_deps} target_wrapper_cuda)
endif()
lite_cc_library(light_api SRCS light_api.cc
        DEPS scope target_wrapper_host model_parser
            ${light_api_deps} ${ops} ${host_kernels} program
        CUDA_DEPS ${cuda_kernels}
        X86_DEPS ${x86_kernels}
        ARM_DEPS ${arm_kernels}
        CV_DEPS paddle_cv_arm
        NPU_DEPS ${npu_kernels}
        APU_DEPS ${apu_kernels}
        XPU_DEPS ${xpu_kernels}
        RKNPU_DEPS ${rknpu_kernels}
        CL_DEPS ${opencl_kernels}
        FPGA_DEPS ${fpga_kernels}
        BM_DEPS ${bm_kernels}
        MLU_DEPS ${mlu_kernels}
        HUAWEI_ASCEND_NPU_DEPS ${huawei_ascend_npu_kernels})

include(ExternalProject)
set(LITE_DEMO_INSTALL_DIR "${THIRD_PARTY_PATH}/inference_demo" CACHE STRING
        "A path setting inference demo download directories.")

if(WITH_TESTING)
    if(NOT WITH_COVERAGE)
        lite_cc_test(test_cxx_api SRCS cxx_api_test.cc
           DEPS cxx_api mir_passes lite_api_test_helper
           ${ops} ${host_kernels}
           X86_DEPS ${x86_kernels}
           CUDA_DEPS ${cuda_kernels}
           ARM_DEPS ${arm_kernels}
           CV_DEPS paddle_cv_arm
           NPU_DEPS ${npu_kernels}
           APU_DEPS ${apu_kernels}
           XPU_DEPS ${xpu_kernels}
           RKNPU_DEPS ${rknpu_kernels}
           CL_DEPS ${opencl_kernels}
           FPGA_DEPS ${fpga_kernels}
           BM_DEPS ${bm_kernels}
           MLU_DEPS ${mlu_kernels}
           HUAWEI_ASCEND_NPU_DEPS ${huawei_ascend_npu_kernels}
           EXCLUDE_COMPILE_DEPS "ON"
           ARGS --model_dir=${LITE_MODEL_DIR}/lite_naive_model
                --optimized_model=${LITE_MODEL_DIR}/lite_naive_model_opt SERIAL)
        add_dependencies(test_cxx_api extern_lite_download_lite_naive_model_tar_gz)
    endif()
    if(NOT LITE_WITH_LIGHT_WEIGHT_FRAMEWORK)
        if(LITE_WITH_X86)
            lite_cc_test(test_googlenet SRCS test_googlenet_lite.cc
               DEPS mir_passes lite_api_test_helper paddle_api_full paddle_api_light gflags utils
               ${ops} ${host_kernels} ${x86_kernels}
               ARGS --model_dir=${LITE_MODEL_DIR}/googlenet)
            add_dependencies(test_googlenet extern_lite_download_GoogleNet_inference_tar_gz)
            lite_cc_test(test_mobilenetv1_lite_x86 SRCS test_mobilenetv1_lite_x86.cc
               DEPS mir_passes lite_api_test_helper paddle_api_full paddle_api_light gflags utils
               ${ops} ${host_kernels} ${x86_kernels}
               ARGS --model_dir=${LITE_MODEL_DIR}/mobilenet_v1)
            add_dependencies(test_mobilenetv1_lite_x86 extern_lite_download_mobilenet_v1_tar_gz)
            lite_cc_test(test_mobilenetv2_lite_x86 SRCS test_mobilenetv2_lite_x86.cc
               DEPS mir_passes lite_api_test_helper paddle_api_full paddle_api_light gflags utils
               ${ops} ${host_kernels} ${x86_kernels}
               ARGS --model_dir=${LITE_MODEL_DIR}/mobilenet_v2_relu)
            add_dependencies(test_mobilenetv2_lite_x86 extern_lite_download_mobilenet_v2_relu_tar_gz)
            lite_cc_test(test_inceptionv4_lite_x86 SRCS test_inceptionv4_lite_x86.cc
               DEPS mir_passes lite_api_test_helper paddle_api_full paddle_api_light gflags utils
               ${ops} ${host_kernels} ${x86_kernels}
               ARGS --model_dir=${LITE_MODEL_DIR}/inception_v4_simple)
            add_dependencies(test_inceptionv4_lite_x86 extern_lite_download_inception_v4_simple_tar_gz)
            lite_cc_test(test_resnet50_lite_x86 SRCS test_resnet50_lite_x86.cc
               DEPS mir_passes lite_api_test_helper paddle_api_full paddle_api_light gflags utils
               ${ops} ${host_kernels} ${x86_kernels}
               ARGS --model_dir=${LITE_MODEL_DIR}/resnet50)
            add_dependencies(test_resnet50_lite_x86 extern_lite_download_resnet50_tar_gz)
            lite_cc_test(test_step_rnn_lite_x86 SRCS test_step_rnn_lite_x86.cc
               DEPS mir_passes lite_api_test_helper paddle_api_full paddle_api_light gflags utils
               ${ops} ${host_kernels} ${x86_kernels}
               ARGS --model_dir=${LITE_MODEL_DIR}/step_rnn)
            add_dependencies(test_step_rnn_lite_x86 extern_lite_download_step_rnn_tar_gz)
        endif()
        if(LITE_WITH_BM)
           lite_cc_test(test_classify_lite_bm SRCS test_classify_lite_bm.cc
              DEPS mir_passes lite_api_test_helper paddle_api_full paddle_api_light gflags utils
              ${ops} ${host_kernels} ${bm_kernels} ${bm_bridges}
              ARGS --model_dir=${LITE_MODEL_DIR}/classify)
           lite_cc_test(test_yolov3_lite_bm SRCS test_yolov3_lite_bm.cc
              DEPS mir_passes lite_api_test_helper paddle_api_full paddle_api_light gflags utils
              ${ops} ${host_kernels} ${bm_kernels} ${bm_bridges}
              ARGS --model_dir=${LITE_MODEL_DIR}/yolov3)
        endif()
    endif()
endif()

if(LITE_WITH_LIGHT_WEIGHT_FRAMEWORK AND WITH_TESTING)
    set(lite_model_test_DEPS cxx_api mir_passes ${ops} ${host_kernels} ${arm_kernels} ${npu_kernels} ${apu_kernels} ${fpga_kernels})

    lite_cc_test(test_mobilenetv1_int8 SRCS mobilenetv1_int8_test.cc
       DEPS ${lite_model_test_DEPS}
       CL_DEPS ${opencl_kernels}
       ARGS --cl_path=${CMAKE_SOURCE_DIR}/lite/backends/opencl
        --model_dir=${LITE_MODEL_DIR}/MobilenetV1_quant SERIAL)
    add_dependencies(test_mobilenetv1_int8 extern_lite_download_MobileNetV1_quant_tar_gz)

    lite_cc_test(test_mobilenetv1 SRCS mobilenetv1_test.cc
       DEPS ${lite_model_test_DEPS}
       CL_DEPS ${opencl_kernels}
       NPU_DEPS ${npu_kernels} ${npu_bridges}
       ARGS --cl_path=${CMAKE_SOURCE_DIR}/lite/backends/opencl
            --model_dir=${LITE_MODEL_DIR}/mobilenet_v1 SERIAL)
    add_dependencies(test_mobilenetv1 extern_lite_download_mobilenet_v1_tar_gz)
    if(NOT WIN32)
       set(LINK_FLAGS "-Wl,--version-script ${PADDLE_SOURCE_DIR}/lite/core/lite.map")
       set_target_properties(test_mobilenetv1 PROPERTIES LINK_FLAGS "${LINK_FLAGS}")
    endif()

    lite_cc_test(test_mobilenetv2 SRCS mobilenetv2_test.cc
       DEPS ${lite_model_test_DEPS}
       CL_DEPS ${opencl_kernels}
       ARGS --cl_path=${CMAKE_SOURCE_DIR}/lite/backends/opencl
            --model_dir=${LITE_MODEL_DIR}/mobilenet_v2_relu SERIAL)
    add_dependencies(test_mobilenetv2 extern_lite_download_mobilenet_v2_relu_tar_gz)
    if(NOT WIN32)
        set_target_properties(test_mobilenetv2 PROPERTIES LINK_FLAGS "${LINK_FLAGS}")
    endif()

    lite_cc_test(test_resnet50 SRCS resnet50_test.cc
       DEPS ${lite_model_test_DEPS} paddle_api_light
       CL_DEPS ${opencl_kernels}
       FPGA_DEPS ${fpga_kernels}
       ARGS --cl_path=${CMAKE_SOURCE_DIR}/lite/backends/opencl
            --model_dir=${LITE_MODEL_DIR}/resnet50 SERIAL)
    add_dependencies(test_resnet50 extern_lite_download_resnet50_tar_gz)

    lite_cc_test(test_resnet50_fpga SRCS resnet50_test_fpga.cc
       DEPS ${lite_model_test_DEPS}
       CL_DEPS ${opencl_kernels}
       FPGA_DEPS ${fpga_kernels})

    lite_cc_test(test_inceptionv4 SRCS inceptionv4_test.cc
       DEPS ${lite_model_test_DEPS}
       CL_DEPS ${opencl_kernels}
       ARGS --cl_path=${CMAKE_SOURCE_DIR}/lite/backends/opencl
            --model_dir=${LITE_MODEL_DIR}/inception_v4 SERIAL)
    add_dependencies(test_inceptionv4 extern_lite_download_inception_v4_simple_tar_gz)

   # brief: we comment ocr_test_ut because we do not supply ocr model to test, it is the reference to infer nlp model
   # lite_cc_test(test_ocr_attention SRCS ocr_attention_test.cc
   #    DEPS ${lite_model_test_DEPS})

   # lite_cc_test(model_run_test_image SRCS model_run_test_image.cc
   #    DEPS ${lite_model_test_DEPS}
   #    CL_DEPS ${opencl_kernels}
   #    FPGA_DEPS ${fpga_kernels})
endif()

lite_cc_library(paddle_api SRCS paddle_api.cc DEPS op_params tensor device_info)

#-----------------------------------------------------------------------------------------------------
# The final inference library for both CxxConfig and MobileConfig.
if (LITE_ON_TINY_PUBLISH)
    lite_cc_library(paddle_api_light SRCS light_api_impl.cc DEPS light_api paddle_api stream)
else()
    lite_cc_library(paddle_api_light SRCS light_api_impl.cc DEPS light_api paddle_api)
endif()
if (NOT LITE_ON_TINY_PUBLISH)
    lite_cc_library(paddle_api_full SRCS cxx_api_impl.cc DEPS cxx_api paddle_api_light
        ${ops}
        ARM_DEPS ${arm_kernels}
        CV_DEPS paddle_cv_arm
        NPU_DEPS ${npu_kernels}
        APU_DEPS ${apu_kernels}
        CL_DEPS ${opencl_kernels}
        FPGA_DEPS ${fpga_kernels}
        BM_DEPS ${bm_kernels}
        HUAWEI_ASCEND_NPU_DEPS ${huawei_ascend_npu_kernels})
    # The final inference library for just MobileConfig.
    bundle_static_library(paddle_api_full paddle_api_full_bundled bundle_full_api)
    target_link_libraries(paddle_api_full ${cuda_deps})
    get_property(fluid_modules GLOBAL PROPERTY FLUID_MODULES)
endif()

bundle_static_library(paddle_api_light paddle_api_light_bundled bundle_light_api)
#-----------------------------------------------------------------------------------------------------

# These tests needs CLI arguments, and is not supported in ARM CI.
# TODO(Superjomn) support latter.
if(NOT WITH_COVERAGE)
    lite_cc_test(test_light_api SRCS light_api_test.cc
        DEPS light_api program mir_passes paddle_api_light
        CL_DEPS ${opencl_kernels}
        FPGA_DEPS ${fpga_kernels}
        RKNPU_DEPS ${rknpu_kernels}
        BM_DEPS ${bm_kernels}
        ARGS --optimized_model=${LITE_MODEL_DIR}/lite_naive_model_opt SERIAL)

    lite_cc_test(test_apis SRCS apis_test.cc
        DEPS cxx_api light_api ${ops} paddle_api_light
        CL_DEPS ${opencl_kernels}
        X86_DEPS ${x86_kernels}
        XPU_DEPS ${xpu_kernels}
        FPGA_DEPS ${fpga_kernels}
        RKNPU_DEPS ${rknpu_kernels}
        BM_DEPS ${bm_kernels}
        MLU_DEPS ${mlu_kernels}
        ARGS --model_dir=${LITE_MODEL_DIR}/lite_naive_model
        --optimized_model=${LITE_MODEL_DIR}/lite_naive_model_opt SERIAL)
endif()

if (LITE_WITH_JAVA AND LITE_WITH_ARM)
    add_subdirectory(android)
endif()

if (LITE_WITH_PYTHON)
    add_subdirectory(python)
    # add library for opt_base
    lite_cc_library(opt_base SRCS opt_base.cc cxx_api_impl.cc paddle_api.cc cxx_api.cc DEPS kernel op optimizer mir_passes utils)
    add_dependencies(opt_base supported_kernel_op_info_h framework_proto all_kernel_faked_cc kernel_list_h)
endif()

if (LITE_ON_TINY_PUBLISH)
    return()
endif()



if (LITE_ON_MODEL_OPTIMIZE_TOOL)
    message(STATUS "Compiling opt")
    lite_cc_binary(opt SRCS opt.cc cxx_api_impl.cc paddle_api.cc cxx_api.cc
        DEPS gflags kernel op optimizer mir_passes utils ${host_kernels})
    add_dependencies(opt op_list_h kernel_list_h all_kernel_faked_cc supported_kernel_op_info_h)
endif(LITE_ON_MODEL_OPTIMIZE_TOOL)

if(NOT WITH_COVERAGE)
    lite_cc_test(test_paddle_api SRCS paddle_api_test.cc DEPS paddle_api_full paddle_api_light
      ${ops}
      ARM_DEPS ${arm_kernels}
      CV_DEPS paddle_cv_arm
      NPU_DEPS ${npu_kernels}
      XPU_DEPS ${xpu_kernels}
      APU_DEPS ${apu_kernels}
      RKNPU_DEPS ${rknpu_kernels}
      CL_DEPS ${opencl_kernels}
      X86_DEPS ${x86_kernels}
      FPGA_DEPS ${fpga_kernels}
      BM_DEPS ${bm_kernels}
      MLU_DEPS ${mlu_kernels}
      HUAWEI_ASCEND_NPU_DEPS ${huawei_ascend_npu_kernels}
      ARGS --model_dir=${LITE_MODEL_DIR}/lite_naive_model SERIAL)
    if (WITH_TESTING)
        add_dependencies(test_paddle_api extern_lite_download_lite_naive_model_tar_gz)
    endif()
endif()

# Some bins
if(NOT IOS)
    lite_cc_binary(test_model_bin SRCS model_test.cc DEPS paddle_api_full paddle_api_light gflags utils
        ${ops} ${host_kernels}
        ARM_DEPS ${arm_kernels}
        CV_DEPS paddle_cv_arm
        NPU_DEPS ${npu_kernels}
        XPU_DEPS ${xpu_kernels}
        MLU_DEPS ${mlu_kernels}
        APU_DEPS ${apu_kernels}
        CL_DEPS ${opencl_kernels}
        BM_DEPS ${bm_kernels}
        RKNPU_DEPS ${rknpu_kernels}
        FPGA_DEPS ${fpga_kernels}
        X86_DEPS ${x86_kernels}
        CUDA_DEPS ${cuda_kernels}
        HUAWEI_ASCEND_NPU_DEPS ${huawei_ascend_npu_kernels})
    
    lite_cc_binary(test_model_detection_bin SRCS model_test_detection.cc DEPS paddle_api_full paddle_api_light gflags utils
        ${ops} ${host_kernels}
        ARM_DEPS ${arm_kernels}
        CV_DEPS paddle_cv_arm
        NPU_DEPS ${npu_kernels}
        XPU_DEPS ${xpu_kernels}
        MLU_DEPS ${mlu_kernels}
        APU_DEPS ${apu_kernels}
        CL_DEPS ${opencl_kernels}
        BM_DEPS ${bm_kernels}
        RKNPU_DEPS ${rknpu_kernels}
        FPGA_DEPS ${fpga_kernels}
        X86_DEPS ${x86_kernels}
        CUDA_DEPS ${cuda_kernels}
        HUAWEI_ASCEND_NPU_DEPS ${huawei_ascend_npu_kernels})
    
    lite_cc_binary(test_model_classify_bin SRCS model_test_classify.cc DEPS paddle_api_full paddle_api_light gflags utils
        ${ops} ${host_kernels}
        ARM_DEPS ${arm_kernels}
        CV_DEPS paddle_cv_arm
        NPU_DEPS ${npu_kernels}
        XPU_DEPS ${xpu_kernels}
        MLU_DEPS ${mlu_kernels}
        APU_DEPS ${apu_kernels}
        CL_DEPS ${opencl_kernels}
        BM_DEPS ${bm_kernels}
        RKNPU_DEPS ${rknpu_kernels}
        FPGA_DEPS ${fpga_kernels}
        X86_DEPS ${x86_kernels}
        CUDA_DEPS ${cuda_kernels}
        HUAWEI_ASCEND_NPU_DEPS ${huawei_ascend_npu_kernels})

    lite_cc_binary(benchmark_bin SRCS benchmark.cc DEPS paddle_api_full paddle_api_light gflags utils
        ${ops} ${host_kernels}
        ARM_DEPS ${arm_kernels}
        CV_DEPS paddle_cv_arm
        NPU_DEPS ${npu_kernels}
        XPU_DEPS ${xpu_kernels}
        RKNPU_DEPS ${rknpu_kernels}
        MLU_DEPS ${mlu_kernels}
        APU_DEPS ${apu_kernels}
        CL_DEPS ${opencl_kernels}
        FPGA_DEPS ${fpga_kernels}
        X86_DEPS ${x86_kernels}
        CUDA_DEPS ${cuda_kernels}
        HUAWEI_ASCEND_NPU_DEPS ${huawei_ascend_npu_kernels})
    
    lite_cc_binary(multithread_test SRCS lite_multithread_test.cc DEPS paddle_api_full paddle_api_light gflags utils
        ${ops} ${host_kernels}
        ARM_DEPS ${arm_kernels}
        CV_DEPS paddle_cv_arm
        NPU_DEPS ${npu_kernels}
        APU_DEPS ${apu_kernels}
        XPU_DEPS ${xpu_kernels}
        RKNPU_DEPS ${rknpu_kernels}
        MLU_DEPS ${mlu_kernels}
        HUAWEI_ASCEND_NPU_DEPS ${huawei_ascend_npu_kernels}
        CL_DEPS ${opencl_kernels}
        BM_DEPS ${bm_kernels}
        FPGA_DEPS ${fpga_kernels}
        X86_DEPS ${x86_kernels}
        CUDA_DEPS ${cuda_kernels})

    lite_cc_binary(test_transformer SRCS transform_test.cc DEPS paddle_api_full paddle_api_light gflags utils
        ${ops} ${host_kernels}
        ARM_DEPS ${arm_kernels}
        CV_DEPS paddle_cv_arm
        NPU_DEPS ${npu_kernels}
        RKNPU_DEPS ${npu_kernels}
        XPU_DEPS ${xpu_kernels}
        APU_DEPS ${apu_kernels}
        CL_DEPS ${opencl_kernels}
        FPGA_DEPS ${fpga_kernels}
        X86_DEPS ${x86_kernels}
        CUDA_DEPS ${cuda_kernels}
        HUAWEI_ASCEND_NPU_DEPS ${huawei_ascend_npu_kernels})
endif()

#lite_cc_binary(cxx_api_bin SRCS cxx_api_bin.cc
    #X86_DEPS operator
    #DEPS light_api model_parser target_wrapper_host mir_passes
    #ARM_DEPS ${arm_kernels}) NPU_DEPS ${npu_kernels})
