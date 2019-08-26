if(LITE_WITH_LIGHT_WEIGHT_FRAMEWORK)
  lite_cc_library(place SRCS paddle_place.cc DEPS logging)
else()
  lite_cc_library(place SRCS paddle_place.cc DEPS glog)
endif(LITE_WITH_LIGHT_WEIGHT_FRAMEWORK)

if (WITH_TESTING)
    lite_cc_library(lite_api_test_helper SRCS lite_api_test_helper.cc
      DEPS scope optimizer target_wrapper_host model_parser program
           ${ops} ${host_kernels}
      CUDA_DEPS kernels_cuda
      X86_DEPS ${x86_kernels})
endif()
if(LITE_WITH_FPGA)
    set(light_api_deps ${light_api_deps} ${fpga_deps})
    set(cxx_api_deps ${cxx_api_deps} ${fpga_deps})
endif()

message(STATUS "get ops ${ops}")
message(STATUS "get X86 kernels ${x86_kernels}")
message(STATUS "get Host kernels ${host_kernels}")
message(STATUS "get ARM kernels ${arm_kernels}")
message(STATUS "get NPU kernels ${npu_kernels}")
message(STATUS "get FPGA kernels ${fpga_kernels}")

# for full api
if (NOT LITE_ON_TINY_PUBLISH)
    set(cxx_api_deps
      scope optimizer target_wrapper_host model_parser program)
    if(LITE_WITH_CUDA)
        set(cxx_api_deps ${cxx_api_deps} kernels_cuda)
        lite_cc_library(cxx_api_cuda SRCS cxx_api.cc DEPS ${cxx_api_deps} target_wrapper_cuda)
        nv_test(test_cxx_api_cuda SRCS cxx_api_test.cc DEPS cxx_api_cuda)
    endif()
    lite_cc_library(cxx_api
                    SRCS cxx_api.cc
                    DEPS ${cxx_api_deps} ${ops} ${host_kernels} program
                    X86_DEPS ${x86_kernels}
                    ARM_DEPS ${arm_kernels}
                    NPU_DEPS ${npu_kernels} ${npu_bridges} npu_pass
                    CL_DEPS ${opencl_kenrels}
                    FPGA_DEPS ${fpga_kenrels})
endif()

# for light api
set(light_api_deps
    scope target_wrapper_host model_parser program)
if(LITE_WITH_CUDA)
    set(light_api_deps ${light_api_deps} target_wrapper_cuda)
endif()
lite_cc_library(light_api SRCS light_api.cc
        DEPS scope target_wrapper_host model_parser
            ${light_api_deps} ${ops} ${host_kernels} program
        CUDA_DEPS target_wrapper_cuda
        X86_DEPS ${x86_kernels}
        ARM_DEPS ${arm_kernels}
        NPU_DEPS ${npu_kernels} ${npu_bridges} npu_pass
        CL_DEPS ${opencl_kenrels}
        FPGA_DEPS ${fpga_kenrels})

include(ExternalProject)
set(LITE_DEMO_INSTALL_DIR "${THIRD_PARTY_PATH}/inference_demo" CACHE STRING
        "A path setting inference demo download directories.")

if(WITH_TESTING)
    lite_cc_test(test_cxx_api SRCS cxx_api_test.cc
       DEPS cxx_api mir_passes lite_api_test_helper
       ${ops} ${host_kernels}
       X86_DEPS ${x86_kernels}
       ARM_DEPS ${arm_kernels}
       NPU_DEPS ${npu_kernels}
       CL_DEPS ${opencl_kernels}
       FPGA_DEPS ${fpga_kernels}
       EXCLUDE_COMPILE_DEPS "ON"
       ARGS --model_dir=${LITE_MODEL_DIR}/lite_naive_model
            --optimized_model=${LITE_MODEL_DIR}/lite_naive_model_opt SERIAL)
    add_dependencies(test_cxx_api extern_lite_download_lite_naive_model_tar_gz)
    if(NOT LITE_WITH_LIGHT_WEIGHT_FRAMEWORK)
        lite_cc_test(test_googlenet SRCS test_googlenet_lite.cc
           DEPS cxx_api mir_passes lite_api_test_helper
           ${ops} ${host_kernels} ${x86_kernels}
           ARGS --model_dir=${LITE_MODEL_DIR}/googlenet)
        add_dependencies(test_googlenet extern_lite_download_GoogleNet_inference_tar_gz)
        lite_cc_test(test_mobilenetv1_lite_x86 SRCS test_mobilenetv1_lite_x86.cc
           DEPS cxx_api mir_passes lite_api_test_helper
           ${ops} ${host_kernels} ${x86_kernels}
           ARGS --model_dir=${LITE_MODEL_DIR}/mobilenet_v1)
        add_dependencies(test_mobilenetv1_lite_x86 extern_lite_download_mobilenet_v1_tar_gz)
        lite_cc_test(test_mobilenetv2_lite_x86 SRCS test_mobilenetv2_lite_x86.cc
           DEPS cxx_api mir_passes lite_api_test_helper
           ${ops} ${host_kernels} ${x86_kernels}
           ARGS --model_dir=${LITE_MODEL_DIR}/mobilenet_v2_relu)
        add_dependencies(test_mobilenetv2_lite_x86 extern_lite_download_mobilenet_v2_relu_tar_gz)
        lite_cc_test(test_inceptionv4_lite_x86 SRCS test_inceptionv4_lite_x86.cc
           DEPS cxx_api mir_passes lite_api_test_helper
           ${ops} ${host_kernels} ${x86_kernels}
           ARGS --model_dir=${LITE_MODEL_DIR}/inception_v4_simple)
        add_dependencies(test_inceptionv4_lite_x86 extern_lite_download_inception_v4_simple_tar_gz)
    endif()
endif()

if(LITE_WITH_LIGHT_WEIGHT_FRAMEWORK AND WITH_TESTING)
    set(lite_model_test_DEPS cxx_api mir_passes ${ops} ${host_kernels} ${arm_kernels} ${npu_kernels} ${fpga_kernels})

    lite_cc_test(test_mobilenetv1_int8 SRCS mobilenetv1_int8_test.cc
       DEPS ${lite_model_test_DEPS}
       CL_DEPS ${opencl_kernels}
       ARGS --cl_path=${CMAKE_SOURCE_DIR}/lite/opencl
        --model_dir=${LITE_MODEL_DIR}/MobilenetV1_quant SERIAL)
    add_dependencies(test_mobilenetv1_int8 extern_lite_download_MobileNetV1_quant_tar_gz)

    lite_cc_test(test_mobilenetv1 SRCS mobilenetv1_test.cc
       DEPS ${lite_model_test_DEPS}
       CL_DEPS ${opencl_kernels}
       NPU_DEPS ${npu_kernels} ${npu_bridges}
       ARGS --cl_path=${CMAKE_SOURCE_DIR}/lite/opencl
            --model_dir=${LITE_MODEL_DIR}/mobilenet_v1 SERIAL)
    add_dependencies(test_mobilenetv1 extern_lite_download_mobilenet_v1_tar_gz)
    set(LINK_FLAGS "-Wl,--version-script ${PADDLE_SOURCE_DIR}/lite/core/lite.map")
    set_target_properties(test_mobilenetv1 PROPERTIES LINK_FLAGS "${LINK_FLAGS}")

    lite_cc_test(test_mobilenetv2 SRCS mobilenetv2_test.cc
       DEPS ${lite_model_test_DEPS}
       CL_DEPS ${opencl_kernels}
       ARGS --cl_path=${CMAKE_SOURCE_DIR}/lite/opencl
            --model_dir=${LITE_MODEL_DIR}/mobilenet_v2_relu SERIAL)
    add_dependencies(test_mobilenetv2 extern_lite_download_mobilenet_v2_relu_tar_gz)
    set_target_properties(test_mobilenetv2 PROPERTIES LINK_FLAGS "${LINK_FLAGS}")

    lite_cc_test(test_resnet50 SRCS resnet50_test.cc
       DEPS ${lite_model_test_DEPS} paddle_api_light
       CL_DEPS ${opencl_kernels}
       FPGA_DEPS ${fpga_kernels}
       ARGS --cl_path=${CMAKE_SOURCE_DIR}/lite/opencl
            --model_dir=${LITE_MODEL_DIR}/resnet50 SERIAL)
    add_dependencies(test_resnet50 extern_lite_download_resnet50_tar_gz)

    lite_cc_test(test_resnet50_fpga SRCS resnet50_test_fpga.cc
       DEPS ${lite_model_test_DEPS}
       CL_DEPS ${opencl_kernels}
       FPGA_DEPS ${fpga_kernels})

    lite_cc_test(test_inceptionv4 SRCS inceptionv4_test.cc
       DEPS ${lite_model_test_DEPS}
       CL_DEPS ${opencl_kernels}
       ARGS --cl_path=${CMAKE_SOURCE_DIR}/lite/opencl
            --model_dir=${LITE_MODEL_DIR}/inception_v4 SERIAL)
    add_dependencies(test_inceptionv4 extern_lite_download_inception_v4_simple_tar_gz)
   # lite_cc_test(test_ocr_attention SRCS ocr_attention_test.cc
   #    DEPS ${lite_model_test_DEPS})

   # lite_cc_test(model_run_test_image SRCS model_run_test_image.cc
   #    DEPS ${lite_model_test_DEPS}
   #    CL_DEPS ${opencl_kernels}
   #    FPGA_DEPS ${fpga_kernels})
endif()

# These tests needs CLI arguments, and is not supported in ARM CI.
# TODO(Superjomn) support latter.
lite_cc_test(test_light_api SRCS light_api_test.cc
    DEPS light_api program mir_passes
    CL_DEPS ${opencl_kernels}
    FPGA_DEPS ${fpga_kernels}
    ARGS --optimized_model=${LITE_MODEL_DIR}/lite_naive_model_opt SERIAL)

lite_cc_test(test_apis SRCS apis_test.cc
  DEPS cxx_api light_api ${ops}
  CL_DEPS ${opencl_kernels}
  X86_DEPS ${x86_kernels}
    FPGA_DEPS ${fpga_kernels}
  ARGS --model_dir=${LITE_MODEL_DIR}/lite_naive_model
       --optimized_model=${LITE_MODEL_DIR}/lite_naive_model_opt SERIAL)

lite_cc_library(paddle_api SRCS paddle_api.cc DEPS op_params tensor)

#-----------------------------------------------------------------------------------------------------
# The final inference library for both CxxConfig and MobileConfig.
if (LITE_ON_TINY_PUBLISH)
    lite_cc_library(paddle_api_light SRCS light_api_impl.cc DEPS light_api paddle_api stream)
else()
    lite_cc_library(paddle_api_light SRCS light_api_impl.cc DEPS light_api paddle_api)
endif()
if (NOT LITE_ON_TINY_PUBLISH)
    lite_cc_library(paddle_api_full SRCS cxx_api_impl.cc DEPS cxx_api paddle_api light_api
      ${ops}
      ARM_DEPS ${arm_kernels}
      NPU_DEPS ${npu_kernels}
      CL_DEPS ${opencl_kernels}
      FPGA_DEPS ${fpga_kernels})
    # The final inference library for just MobileConfig.
    bundle_static_library(paddle_api_full paddle_api_full_bundled bundle_full_api)
endif()
bundle_static_library(paddle_api_light paddle_api_light_bundled bundle_light_api)
#-----------------------------------------------------------------------------------------------------

if (LITE_WITH_JAVA AND LITE_WITH_ARM)
    add_subdirectory(android)
endif()

if (LITE_ON_TINY_PUBLISH)
    return()
endif()
lite_cc_test(test_paddle_api SRCS paddle_api_test.cc DEPS paddle_api_full paddle_api_light
  ${ops}
  ARM_DEPS ${arm_kernels}
  NPU_DEPS ${npu_kernels}
  CL_DEPS ${opencl_kernels}
  X86_DEPS ${x86_kernels}
  FPGA_DEPS ${fpga_kernels}
  ARGS --model_dir=${LITE_MODEL_DIR}/lite_naive_model SERIAL)
if (WITH_TESTING)
    add_dependencies(test_paddle_api extern_lite_download_lite_naive_model_tar_gz)
endif()

# Some bins
if(NOT IOS)
  lite_cc_binary(test_model_bin SRCS model_test.cc DEPS paddle_api_full paddle_api_light gflags
    ${ops}
    ARM_DEPS ${arm_kernels}
    NPU_DEPS ${npu_kernels}
    CL_DEPS ${opencl_kernels}
    FPGA_DEPS ${fpga_kernels}
    X86_DEPS ${x86_kernels})
  lite_cc_binary(benchmark_bin SRCS benchmark.cc DEPS paddle_api_full paddle_api_light gflags
    ${ops}
    ARM_DEPS ${arm_kernels}
    NPU_DEPS ${npu_kernels}
    CL_DEPS ${opencl_kernels}
    FPGA_DEPS ${fpga_kernels}
    X86_DEPS ${x86_kernels})
endif()

#lite_cc_binary(cxx_api_bin SRCS cxx_api_bin.cc
    #X86_DEPS operator
    #DEPS light_api model_parser target_wrapper_host mir_passes
    #ARM_DEPS ${arm_kernels}) NPU_DEPS ${npu_kernels})

lite_cc_binary(model_optimize_tool SRCS model_optimize_tool.cc
    DEPS paddle_api_full gflags
    CL_DEPS ${opencl_kernels})
