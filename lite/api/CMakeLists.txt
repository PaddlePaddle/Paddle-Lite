if(MSVC)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /bigobj")
endif()

#----------------------------------------------- NOT CHANGE ---------------------------------------
# A trick to generate the paddle_use_kernels.h
execute_process(
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/lite/api/paddle_lite_factory_helper.h ${CMAKE_BINARY_DIR}/lite/api/)

execute_process(
    COMMAND python ${CMAKE_SOURCE_DIR}/lite/tools/cmake_tools/parse_kernel_registry.py
    ${kernels_src_list}
    ${fake_kernels_src_list}
    ${CMAKE_BINARY_DIR}/lite/api/paddle_use_kernels.h
    "${LITE_OPTMODEL_DIR}/.tailored_kernels_list"
    ${LITE_BUILD_TAILOR}
    ${LITE_BUILD_EXTRA}
    ${LITE_WITH_ARM82_FP16}
    RESULT_VARIABLE result)
# A trick to generate the paddle_use_ops.h
execute_process(
    COMMAND python ${CMAKE_SOURCE_DIR}/lite/tools/cmake_tools/parse_op_registry.py
    ${ops_src_list}
    ${CMAKE_BINARY_DIR}/lite/api/paddle_use_ops.h
    "${LITE_OPTMODEL_DIR}/.tailored_ops_list"
    ${LITE_BUILD_TAILOR}
    ${LITE_BUILD_EXTRA}
    RESULT_VARIABLE result)






if(LITE_WITH_LIGHT_WEIGHT_FRAMEWORK OR (NOT LITE_WITH_LOG))
  lite_cc_library(place SRCS paddle_place.cc DEPS logging)
else()
  lite_cc_library(place SRCS paddle_place.cc DEPS glog)
endif()

if (LITE_ON_TINY_PUBLISH)
    set(CMAKE_CXX_FLAGS_RELEASE "-Os -DNDEBUG")
    set(CMAKE_C_FLAGS_RELEASE "-Os -DNDEBUG")
endif()

set(light_lib_DEPS light_api paddle_api paddle_api_light utils core kernels CACHE INTERNAL "")
set(full_lib_DEPS paddle_api paddle_api_light  paddle_api_full framework_proto core utils kernels CACHE INTERNAL "")
set(external_libs_DEPS "" CACHE INTERNAL "")
if ((NOT LITE_ON_TINY_PUBLISH) AND (LITE_WITH_CUDA OR LITE_WITH_X86 OR LITE_WITH_SW OR LITE_WITH_BM OR LITE_WITH_HUAWEI_ASCEND_NPU OR ARM_TARGET_OS STREQUAL "android" OR ARM_TARGET_OS STREQUAL "armlinux"))
    #full api dynamic library
    if(LITE_WITH_X86)
        set(external_libs_DEPS ${external_libs_DEPS} xxhash CACHE INTERNAL "")
        if ((NOT LITE_ON_MODEL_OPTIMIZE_TOOL) AND WITH_MKL)
            if(WITH_STATIC_MKL)
                set(external_libs_DEPS ${external_libs_DEPS} ${MKLML_LIBRARIES} CACHE INTERNAL "")
            else()
                set(external_libs_DEPS ${external_libs_DEPS} mklml CACHE INTERNAL "")
            endif()
        endif ()
     endif()


    lite_cc_library(paddle_full_api_shared SHARED SRCS paddle_api.cc light_api.cc cxx_api.cc cxx_api_impl.cc light_api_impl.cc
                  DEPS ${full_lib_DEPS} ${external_libs_DEPS})
    add_dependencies(paddle_full_api_shared fbs_headers)

    if(LITE_WITH_CUDA)
        target_link_libraries(paddle_full_api_shared ${math_cuda} "-Wl,--whole-archive" kernels "-Wl,--no-whole-archive")
    endif(LITE_WITH_CUDA)
    if(LITE_WITH_CV)
      target_link_libraries(paddle_full_api_shared "-Wl,--whole-archive" paddle_cv_arm "-Wl,--no-whole-archive")
    endif(LITE_WITH_CV)

    #light api dynamic library
    lite_cc_library(paddle_light_api_shared SHARED SRCS paddle_api.cc light_api.cc light_api_impl.cc
                  DEPS ${light_lib_DEPS} ${external_libs_DEPS}
                  CV_DEPS paddle_cv_arm
                  )

    if(WIN32)
        target_link_libraries(paddle_full_api_shared shlwapi.lib)
        target_link_libraries(paddle_light_api_shared shlwapi.lib)
    endif()
    target_link_libraries(paddle_light_api_shared ${light_lib_DEPS} kernels)
   if(${HOST_SYSTEM} MATCHES "macosx" AND NOT (ARM_TARGET_OS STREQUAL "android"))
        set(LINK_MAP_FILE "${PADDLE_SOURCE_DIR}/lite/core/exported_symbols.lds")
        set(LINK_FLAGS "-Wl,-exported_symbols_list ${LINK_MAP_FILE}")
        add_custom_command(OUTPUT ${LINK_MAP_FILE} COMMAND :)
        add_custom_target(custom_linker_map DEPENDS ${LINK_MAP_FILE})
        set_target_properties(paddle_full_api_shared PROPERTIES LINK_FLAGS ${LINK_FLAGS})
        add_dependencies(paddle_full_api_shared custom_linker_map)
        set_target_properties(paddle_light_api_shared PROPERTIES LINK_FLAGS ${LINK_FLAGS})
        add_dependencies(paddle_light_api_shared custom_linker_map)
   elseif(NOT WIN32)
        set(LINK_MAP_FILE "${PADDLE_SOURCE_DIR}/lite/core/lite.map")
        set(LINK_FLAGS "-Wl,--version-script ${LINK_MAP_FILE}")
        add_custom_command(OUTPUT ${LINK_MAP_FILE} COMMAND :)
        add_custom_target(custom_linker_map DEPENDS ${LINK_MAP_FILE})
        set_target_properties(paddle_full_api_shared PROPERTIES LINK_FLAGS ${LINK_FLAGS})
        add_dependencies(paddle_full_api_shared custom_linker_map)
        set_target_properties(paddle_light_api_shared PROPERTIES LINK_FLAGS ${LINK_FLAGS})
        add_dependencies(paddle_light_api_shared custom_linker_map)
   endif()
   if(NOT WIN32 AND NOT ARM_TARGET_OS STREQUAL "android" AND WITH_TESTING)
        # check symbol hidden
        FILE(WRITE ${CMAKE_CURRENT_BINARY_DIR}/check_symbol.cmake
            "execute_process(COMMAND sh -c \"${CMAKE_CURRENT_SOURCE_DIR}/../tools/check_symbol.sh"
            " ${CMAKE_CURRENT_BINARY_DIR}/libpaddle_light_api_shared.so\" RESULT_VARIABLE symbol_res)\n"
            "if(NOT \"\${symbol_res}\" STREQUAL \"0\")\n"
            "  message(FATAL_ERROR \"Check libpaddle_light_api_shared.so symbol failed.\")\n"
            "endif()\n"
            "execute_process(COMMAND sh -c \"${CMAKE_CURRENT_SOURCE_DIR}/../tools/check_symbol.sh"
            " ${CMAKE_CURRENT_BINARY_DIR}/libpaddle_full_api_shared.so\" RESULT_VARIABLE symbol_res)\n"
            "if(NOT \"\${symbol_res}\" STREQUAL \"0\")\n"
            "  message(FATAL_ERROR \"Check libpaddle_full_api_shared.so symbol failed.\")\n"
            "endif()\n")
        add_custom_command(
            OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/.check_symbol"
            COMMAND ${CMAKE_COMMAND} -P "${CMAKE_CURRENT_BINARY_DIR}/check_symbol.cmake"
            DEPENDS paddle_light_api_shared paddle_full_api_shared )
        add_custom_target(check_symbol ALL DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/.check_symbol")
        add_dependencies(lite_compile_deps check_symbol)
    endif()
else()
    # Compiling steps in tiny_publish format:
    # 1. compile all source files into .object `PADDLELITE_OBJS`
    add_library(PADDLELITE_OBJS OBJECT ${__lite_cc_files} paddle_api.cc light_api.cc light_api_impl.cc)
    add_dependencies(PADDLELITE_OBJS fbs_headers)
    if (IOS)
        # only sttaic lib is produced for IOS platform.
        add_library(paddle_api_light_bundled STATIC $<TARGET_OBJECTS:PADDLELITE_OBJS>)
    else()
        # 1. enable -flto compiling flag if toochain==gcc
        # TODO (hong19860320): Disable lto temporarily since it causes fail to catch the exceptions in android when toolchain is gcc.
        if (NOT (ARM_TARGET_LANG STREQUAL "clang"))
            if (ARM_TARGET_OS STREQUAL "android" AND LITE_WITH_EXCEPTION)
                set(TARGET_COMIPILE_FLAGS "")
            else()
                set(TARGET_COMIPILE_FLAGS "-fdata-sections -flto")
            endif()
        else()
            set(TARGET_COMIPILE_FLAGS "-fdata-sections")
        endif()
        #   1.1 enable -flto on PADDLELITE_OBJS
        set_target_properties(PADDLELITE_OBJS PROPERTIES COMPILE_FLAGS "${TARGET_COMIPILE_FLAGS}")

        # 2. produce dynamic lib from PADDLELITE_OBJS
        add_library(paddle_light_api_shared SHARED $<TARGET_OBJECTS:PADDLELITE_OBJS>)
        set_target_properties(paddle_light_api_shared PROPERTIES COMPILE_FLAGS "${TARGET_COMIPILE_FLAGS}")
        #   2.1 link `paddle_light_api_shared` to third-party libs
        if (LITE_WITH_NPU)
            # Need to add HIAI runtime libs (libhiai.so) dependency
            target_link_libraries(paddle_light_api_shared ${npu_builder_libs} ${npu_runtime_libs})
        endif()
        if (LITE_WITH_RKNPU)
            # Need to add RKNPU runtime libs dependency
            target_link_libraries(paddle_light_api_shared ${rknpu_builder_libs} ${rknpu_runtime_libs})
        endif()
        if (LITE_WITH_IMAGINATION_NNA)
            # Need to add IMG IMAGINATION_NNA runtime libs (libimgdnn.so, libnnasession.so) dependency
            #target_link_libraries(paddle_light_api_shared ${nna_builder_libs} ${nna_runtime_libs})
        endif()

        # 3. produce java lib from `PADDLELITE_OBJS` if LITE_WITH_JAVA=ON
        if (LITE_WITH_JAVA)
          add_library(paddle_lite_jni SHARED $<TARGET_OBJECTS:PADDLELITE_OBJS> android/jni/native/paddle_lite_jni.cc android/jni/native/tensor_jni.cc)
          set_target_properties(paddle_lite_jni PROPERTIES COMPILE_FLAGS "${TARGET_COMIPILE_FLAGS}")
          set_target_properties(paddle_lite_jni PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lite/api/android/jni/native)
          if (LITE_WITH_NPU)
              # Need to add HIAI runtime libs (libhiai.so) dependency
              target_link_libraries(paddle_lite_jni ${npu_builder_libs} ${npu_runtime_libs})
          endif()
        endif()

        # 4. produce static lib `libpaddle_api_light_bundled.a` from `PADDLELITE_OBJS`
        # '-flto' is only supported by dynamic lib
        if(TARGET_COMIPILE_FLAGS MATCHES ".*-flto.*")
            add_library(paddle_api_light_bundled STATIC ${__lite_cc_files} paddle_api.cc light_api.cc light_api_impl.cc)
        else()
            add_library(paddle_api_light_bundled STATIC $<TARGET_OBJECTS:PADDLELITE_OBJS>)
        endif()
    endif()
endif()

if (WITH_TESTING)
    lite_cc_library(lite_api_test_helper SRCS lite_api_test_helper.cc
      DEPS core target_wrapper_host model_parser
           ops kernels)
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

if(LITE_WITH_IMAGINATION_NNA)
    set(light_api_deps ${light_api_deps} ${imagination_nna_deps})
    set(cxx_api_deps ${cxx_api_deps} ${imagination_nna_deps})
endif()

if(LITE_WITH_INTEL_FPGA)
    set(light_api_deps ${light_api_deps} ${intel_fpga_deps})
    set(cxx_api_deps ${cxx_api_deps} ${intel_fpga_deps})
endif()

if(LITE_WITH_HUAWEI_ASCEND_NPU)
    set(light_api_deps ${light_api_deps} ${huawei_ascend_npu_deps})
    set(cxx_api_deps ${cxx_api_deps} ${huawei_ascend_npu_deps})
endif()

if(LITE_WITH_NNADAPTER)
    set(light_api_deps ${light_api_deps} ${nnadapter_deps})
    set(cxx_api_deps ${cxx_api_deps} ${nnadapter_deps})
endif()

list(LENGTH ops num_ops)

# for full api
if (NOT LITE_ON_TINY_PUBLISH)
    set(cxx_api_deps
    target_wrapper_host model_parser core)
    lite_cc_library(cxx_api
                        SRCS cxx_api.cc
                        DEPS ${cxx_api_deps} ops
                        CV_DEPS paddle_cv_arm)
endif()

# for light api
set(light_api_deps
    target_wrapper_host model_parser core)
if(LITE_WITH_CUDA)
    get_property(cuda_deps GLOBAL PROPERTY CUDA_MODULES)
    set(light_api_deps ${light_api_deps} target_wrapper_cuda)
endif()
lite_cc_library(light_api SRCS light_api.cc
        DEPS core target_wrapper_host
            ${light_api_deps} ops
        CV_DEPS paddle_cv_arm)

include(ExternalProject)
set(LITE_DEMO_INSTALL_DIR "${THIRD_PARTY_PATH}/inference_demo" CACHE STRING
        "A path setting inference demo download directories.")

if(WITH_TESTING)
    if(NOT WITH_COVERAGE)
        lite_cc_test(test_cxx_api SRCS cxx_api_test.cc
           DEPS cxx_api lite_api_test_helper
           ops
           EXCLUDE_COMPILE_DEPS "ON"
           ARGS --model_dir=${LITE_MODEL_DIR}/lite_naive_model
                --optimized_model=${LITE_MODEL_DIR}/lite_naive_model_opt SERIAL)
        add_dependencies(test_cxx_api extern_lite_download_lite_naive_model_tar_gz)
    endif()
    if(NOT LITE_WITH_LIGHT_WEIGHT_FRAMEWORK)
        if(LITE_WITH_X86)
            lite_cc_test(test_googlenet SRCS test_googlenet_lite.cc
               DEPS lite_api_test_helper paddle_api_full paddle_api_light gflags utils
               ops kernels
               ARGS --model_dir=${LITE_MODEL_DIR}/googlenet)
            add_dependencies(test_googlenet extern_lite_download_GoogleNet_inference_tar_gz)
            lite_cc_test(test_mobilenetv1_lite_x86 SRCS test_mobilenetv1_lite_x86.cc
               DEPS lite_api_test_helper paddle_api_full paddle_api_light gflags utils
               ops kernels
               ARGS --model_dir=${LITE_MODEL_DIR}/mobilenet_v1)
            if(LITE_WITH_METAL)
                lite_cc_test(test_mobilenetv1_lite_metal SRCS test_mobilenetv1_lite_metal.cc
                        DEPS lite_api_test_helper paddle_api_full paddle_api_light gflags utils
                        ops kernels
                        ARGS --model_dir=${LITE_MODEL_DIR}/mobilenet_v1 --metal_dir=${CMAKE_BINARY_DIR}/lite/backends/metal/lite.metallib)
            endif()
            add_dependencies(test_mobilenetv1_lite_x86 extern_lite_download_mobilenet_v1_tar_gz)
            lite_cc_test(test_mobilenetv2_lite_x86 SRCS test_mobilenetv2_lite_x86.cc
               DEPS lite_api_test_helper paddle_api_full paddle_api_light gflags utils
               ops kernels
               ARGS --model_dir=${LITE_MODEL_DIR}/mobilenet_v2_relu)
            add_dependencies(test_mobilenetv2_lite_x86 extern_lite_download_mobilenet_v2_relu_tar_gz)
            lite_cc_test(test_inceptionv4_lite_x86 SRCS test_inceptionv4_lite_x86.cc
               DEPS lite_api_test_helper paddle_api_full paddle_api_light gflags utils
               ops kernels
               ARGS --model_dir=${LITE_MODEL_DIR}/inception_v4_simple)
            add_dependencies(test_inceptionv4_lite_x86 extern_lite_download_inception_v4_simple_tar_gz)
            lite_cc_test(test_resnet50_lite_x86 SRCS test_resnet50_lite_x86.cc
               DEPS lite_api_test_helper paddle_api_full paddle_api_light gflags utils
               ops kernels
               ARGS --model_dir=${LITE_MODEL_DIR}/resnet50)
            add_dependencies(test_resnet50_lite_x86 extern_lite_download_resnet50_tar_gz)
            lite_cc_test(test_step_rnn_lite_x86 SRCS test_step_rnn_lite_x86.cc
               DEPS lite_api_test_helper paddle_api_full paddle_api_light gflags utils
               ops kernels
               ARGS --model_dir=${LITE_MODEL_DIR}/step_rnn)
            add_dependencies(test_step_rnn_lite_x86 extern_lite_download_step_rnn_tar_gz)
        endif()
        if(LITE_WITH_BM)
           lite_cc_test(test_classify_lite_bm SRCS test_classify_lite_bm.cc
              DEPS lite_api_test_helper paddle_api_full paddle_api_light gflags utils
              ops kernels ${bm_bridges}
              ARGS --model_dir=${LITE_MODEL_DIR}/classify)
           lite_cc_test(test_yolov3_lite_bm SRCS test_yolov3_lite_bm.cc
              DEPS lite_api_test_helper paddle_api_full paddle_api_light gflags utils
              ops kernels ${bm_bridges}
              ARGS --model_dir=${LITE_MODEL_DIR}/yolov3)
        endif()
    endif()
endif()

if(LITE_WITH_LIGHT_WEIGHT_FRAMEWORK AND WITH_TESTING)
    set(lite_model_test_DEPS cxx_api ops kernels)

    lite_cc_test(test_mobilenetv1_int8 SRCS mobilenetv1_int8_test.cc
       DEPS ${lite_model_test_DEPS}
       ARGS --cl_path=${CMAKE_SOURCE_DIR}/lite/backends/opencl
        --model_dir=${LITE_MODEL_DIR}/MobilenetV1_quant SERIAL)
    add_dependencies(test_mobilenetv1_int8 extern_lite_download_MobileNetV1_quant_tar_gz)

    lite_cc_test(test_mobilenetv1 SRCS mobilenetv1_test.cc
       DEPS ${lite_model_test_DEPS} ${npu_bridges}
       ARGS --cl_path=${CMAKE_SOURCE_DIR}/lite/backends/opencl
            --model_dir=${LITE_MODEL_DIR}/mobilenet_v1 SERIAL)
    add_dependencies(test_mobilenetv1 extern_lite_download_mobilenet_v1_tar_gz)
    if(NOT WIN32)
       set(LINK_FLAGS "-Wl,--version-script ${PADDLE_SOURCE_DIR}/lite/core/lite.map")
       set_target_properties(test_mobilenetv1 PROPERTIES LINK_FLAGS "${LINK_FLAGS}")
    endif()
    
    lite_cc_test(test_mobilenetv1_int16 SRCS mobilenetv1_int16_test.cc
       DEPS ${lite_model_test_DEPS} ${light_lib_DEPS}
       NPU_DEPS ${npu_bridges}
       ARGS --cl_path=${CMAKE_SOURCE_DIR}/lite/backends/opencl
            --model_dir=${LITE_MODEL_DIR}/mobilenet_v1_int16 SERIAL)
    add_dependencies(test_mobilenetv1 extern_lite_download_mobilenet_v1_int16_tar_gz)
    
    lite_cc_test(test_mobilenetv1_opt_quant SRCS mobilenetv1_opt_quant_test.cc
       DEPS ${lite_model_test_DEPS} ${light_lib_DEPS} paddle_api_full
       ARGS --model_dir=${LITE_MODEL_DIR}/mobilenet_v1 SERIAL)
    add_dependencies(test_mobilenetv1 extern_lite_download_mobilenet_v1_tar_gz)

    lite_cc_test(test_mobilenetv2 SRCS mobilenetv2_test.cc
       DEPS ${lite_model_test_DEPS}
       ARGS --cl_path=${CMAKE_SOURCE_DIR}/lite/backends/opencl
            --model_dir=${LITE_MODEL_DIR}/mobilenet_v2_relu SERIAL)
    add_dependencies(test_mobilenetv2 extern_lite_download_mobilenet_v2_relu_tar_gz)
    if(NOT WIN32)
        set_target_properties(test_mobilenetv2 PROPERTIES LINK_FLAGS "${LINK_FLAGS}")
    endif()

    lite_cc_test(test_resnet50 SRCS resnet50_test.cc
       DEPS ${lite_model_test_DEPS} paddle_api_light
       ARGS --cl_path=${CMAKE_SOURCE_DIR}/lite/backends/opencl
            --model_dir=${LITE_MODEL_DIR}/resnet50 SERIAL)
    add_dependencies(test_resnet50 extern_lite_download_resnet50_tar_gz)

    lite_cc_test(test_resnet50_fpga SRCS resnet50_test_fpga.cc
       DEPS ${lite_model_test_DEPS})

    lite_cc_test(test_inceptionv4 SRCS inceptionv4_test.cc
       DEPS ${lite_model_test_DEPS}
       ARGS --cl_path=${CMAKE_SOURCE_DIR}/lite/backends/opencl
            --model_dir=${LITE_MODEL_DIR}/inception_v4 SERIAL)
    add_dependencies(test_inceptionv4 extern_lite_download_inception_v4_simple_tar_gz)

    lite_cc_test(test_fast_rcnn SRCS fast_rcnn_test.cc
       DEPS ${lite_model_test_DEPS} ${light_lib_DEPS} paddle_api_full
       ARGS --cl_path=${CMAKE_SOURCE_DIR}/lite/backends/opencl)
    add_dependencies(test_fast_rcnn extern_lite_download_fast_rcnn_fluid184_tar_gz)

   # brief: we comment ocr_test_ut because we do not supply ocr model to test, it is the reference to infer nlp model
   # lite_cc_test(test_ocr_attention SRCS ocr_attention_test.cc
   #    DEPS ${lite_model_test_DEPS})

   # lite_cc_test(model_run_test_image SRCS model_run_test_image.cc
   #    DEPS ${lite_model_test_DEPS})

endif()


lite_cc_library(paddle_api SRCS paddle_api.cc DEPS ops core)

#-----------------------------------------------------------------------------------------------------
# The final inference library for both CxxConfig and MobileConfig.
if (LITE_ON_TINY_PUBLISH)
    lite_cc_library(paddle_api_light SRCS light_api_impl.cc DEPS light_api paddle_api stream)
else()
    lite_cc_library(paddle_api_light SRCS light_api_impl.cc DEPS light_api paddle_api core utils kernels ops)
endif()
if (NOT LITE_ON_TINY_PUBLISH)
    lite_cc_library(paddle_api_full SRCS cxx_api_impl.cc DEPS cxx_api paddle_api_light
        ops paddle_api framework_proto core utils kernels
        CV_DEPS paddle_cv_arm)
    # The final inference library for just MobileConfig.
    bundle_static_library(paddle_api_full paddle_api_full_bundled bundle_full_api)
    target_link_libraries(paddle_api_full ${cuda_deps})
    get_property(fluid_modules GLOBAL PROPERTY FLUID_MODULES)
    bundle_static_library(paddle_api_light paddle_api_light_bundled bundle_light_api)
endif()

#-----------------------------------------------------------------------------------------------------

# These tests needs CLI arguments, and is not supported in ARM CI.
# TODO(Superjomn) support latter.
if(NOT WITH_COVERAGE)
    lite_cc_test(test_light_api SRCS light_api_test.cc
        DEPS paddle_api_full paddle_api_light ops
        ARGS --optimized_model=${LITE_MODEL_DIR}/lite_naive_model_opt SERIAL)

    lite_cc_test(test_apis SRCS apis_test.cc
        DEPS paddle_api_full paddle_api_light ops kernels
        ARGS --model_dir=${LITE_MODEL_DIR}/lite_naive_model
        --optimized_model=${LITE_MODEL_DIR}/lite_naive_model_opt SERIAL)
endif()

if (LITE_WITH_JAVA AND LITE_WITH_ARM)
    add_subdirectory(android)
endif()

if (LITE_WITH_PYTHON)
    add_subdirectory(python)
    # add library for opt_base
    lite_cc_library(opt_base SRCS tools/opt_base.cc cxx_api_impl.cc paddle_api.cc cxx_api.cc
        DEPS core utils model_parser)
    add_dependencies(opt_base framework_proto kernels)
endif()

if (LITE_ON_TINY_PUBLISH)
    return()
endif()



if (LITE_ON_MODEL_OPTIMIZE_TOOL)
    message(STATUS "Compiling opt")
    lite_cc_binary(opt SRCS tools/opt.cc tools/opt_base.cc cxx_api_impl.cc paddle_api.cc cxx_api.cc
        DEPS gflags utils kernels core ops model_parser)
endif(LITE_ON_MODEL_OPTIMIZE_TOOL)

if(NOT WITH_COVERAGE)
    lite_cc_test(test_paddle_api SRCS paddle_api_test.cc DEPS paddle_api_full paddle_api_light
      ops kernels
      CV_DEPS paddle_cv_arm
      ARGS --model_dir=${LITE_MODEL_DIR}/lite_naive_model SERIAL)
    if (WITH_TESTING)
        add_dependencies(test_paddle_api extern_lite_download_lite_naive_model_tar_gz)
    endif()
endif()

# Some bins
if(NOT IOS)
    lite_cc_binary(test_model_bin SRCS tools/model_test.cc DEPS paddle_api_full paddle_api_light gflags utils core
        ops kernels ${full_lib_DEPS} ${external_libs_DEPS}
        CV_DEPS paddle_cv_arm)
    
    lite_cc_binary(test_model_detection_bin SRCS model_test_detection.cc DEPS paddle_api_full paddle_api_light gflags utils core
        ops kernels ${external_libs_DEPS}
        CV_DEPS paddle_cv_arm)
    
    lite_cc_binary(test_model_classify_bin SRCS model_test_classify.cc DEPS paddle_api_full paddle_api_light gflags utils core
        ops kernels ${external_libs_DEPS}
        CV_DEPS paddle_cv_arm)

    lite_cc_binary(benchmark_bin SRCS tools/benchmark.cc DEPS paddle_api_full paddle_api_light gflags utils core
        ops kernels ${external_libs_DEPS}
        CV_DEPS paddle_cv_arm)
    
    lite_cc_binary(multithread_test SRCS lite_multithread_test.cc DEPS paddle_api_full paddle_api_light gflags utils core
        ops kernels
        CV_DEPS paddle_cv_arm)

    lite_cc_binary(test_transformer SRCS transform_test.cc DEPS paddle_api_full paddle_api_light gflags utils core
        ops ${external_libs_DEPS} kernels
        CV_DEPS paddle_cv_arm)
endif()

#lite_cc_binary(cxx_api_bin SRCS cxx_api_bin.cc
    #X86_DEPS operator
    #DEPS light_api model_parser target_wrapper_host kernels)
