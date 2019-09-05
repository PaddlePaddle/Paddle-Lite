include(lite)

message(WARNING "Lite enabled!")
message(STATUS "LIGHT_FRAMEWORK:\t${LITE_WITH_LIGHT_WEIGHT_FRAMEWORK}")
message(STATUS "LITE_WITH_CUDA:\t${LITE_WITH_CUDA}")
message(STATUS "LITE_WITH_X86:\t${LITE_WITH_X86}")
message(STATUS "LITE_WITH_ARM:\t${LITE_WITH_ARM}")
message(STATUS "LITE_WITH_NPU:\t${LITE_WITH_NPU}")
message(STATUS "LITE_WITH_FPGA:\t${LITE_WITH_FPGA}")
message(STATUS "LITE_WITH_PROFILE:\t${LITE_WITH_PROFILE}")

set(LITE_MODEL_DIR "${THIRD_PARTY_PATH}/install")
set(LITE_ON_MOBILE ${LITE_WITH_LIGHT_WEIGHT_FRAMEWORK})

add_subdirectory(utils)
add_subdirectory(operators)
add_subdirectory(kernels)
add_subdirectory(core)
add_subdirectory(model_parser)
add_subdirectory(api)
add_subdirectory(fluid)
add_subdirectory(backends)

if (NOT LITE_ON_TINY_PUBLISH)
    add_subdirectory(tests)
    add_subdirectory(tools)
endif()
if (LITE_WITH_LIGHT_WEIGHT_FRAMEWORK AND NOT LITE_ON_TINY_PUBLISH)
  add_subdirectory(gen_code)
endif()

if (WITH_TESTING)
    lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL} "lite_naive_model.tar.gz")
    if(LITE_WITH_LIGHT_WEIGHT_FRAMEWORK)
	lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL} "mobilenet_v1.tar.gz")
        lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL} "mobilenet_v2_relu.tar.gz")
        lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL} "resnet50.tar.gz")
        lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL} "inception_v4_simple.tar.gz")
        lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL} "MobileNetV1_quant.tar.gz")
    endif()
    if(NOT LITE_WITH_LIGHT_WEIGHT_FRAMEWORK)
        lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL} "GoogleNet_inference.tar.gz")
	      lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL} "mobilenet_v1.tar.gz")
        lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL} "mobilenet_v2_relu.tar.gz")
        lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL} "resnet50.tar.gz")
        lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL} "inception_v4_simple.tar.gz")
    endif()
endif()

if (LITE_WITH_LIGHT_WEIGHT_FRAMEWORK AND LITE_WITH_ARM)
    # for publish
    set(INFER_LITE_PUBLISH_ROOT "${CMAKE_BINARY_DIR}/inference_lite_lib.${ARM_TARGET_OS}.${ARM_TARGET_ARCH_ABI}")
    if (LITE_WITH_OPENCL)
        set(INFER_LITE_PUBLISH_ROOT "${INFER_LITE_PUBLISH_ROOT}.opencl")
    endif(LITE_WITH_OPENCL)
    if (LITE_WITH_NPU)
        set(INFER_LITE_PUBLISH_ROOT "${INFER_LITE_PUBLISH_ROOT}.npu")
    endif(LITE_WITH_NPU)
    message(STATUS "publish inference lib to ${INFER_LITE_PUBLISH_ROOT}")

    # The final target for publish lite lib
    add_custom_target(publish_inference)
    if (NOT LITE_ON_TINY_PUBLISH)
        # add cxx lib
        add_custom_target(publish_inference_cxx_lib ${TARGET}
                COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/cxx/lib"
                COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/bin"
                COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/cxx/include"
                COMMAND cp "${CMAKE_SOURCE_DIR}/lite/api/paddle_*.h" "${INFER_LITE_PUBLISH_ROOT}/cxx/include"
                COMMAND cp "${CMAKE_BINARY_DIR}/libpaddle_api_full_bundled.a" "${INFER_LITE_PUBLISH_ROOT}/cxx/lib"
                COMMAND cp "${CMAKE_BINARY_DIR}/libpaddle_api_light_bundled.a" "${INFER_LITE_PUBLISH_ROOT}/cxx/lib"
                COMMAND cp "${CMAKE_BINARY_DIR}/lite/api/model_optimize_tool" "${INFER_LITE_PUBLISH_ROOT}/bin"
                COMMAND cp "${CMAKE_BINARY_DIR}/lite/gen_code/paddle_code_generator" "${INFER_LITE_PUBLISH_ROOT}/bin"
                COMMAND cp "${CMAKE_BINARY_DIR}/lite/api/test_model_bin" "${INFER_LITE_PUBLISH_ROOT}/bin"
                )
            if(NOT IOS)
                add_dependencies(publish_inference_cxx_lib model_optimize_tool)
                add_dependencies(publish_inference_cxx_lib paddle_code_generator)
                add_dependencies(publish_inference_cxx_lib bundle_full_api)
                add_dependencies(publish_inference_cxx_lib bundle_light_api)
                add_dependencies(publish_inference_cxx_lib test_model_bin)
                add_dependencies(publish_inference publish_inference_cxx_lib)
                add_custom_command(TARGET publish_inference_cxx_lib POST_BUILD
                        COMMAND ${CMAKE_STRIP} "--strip-debug" ${INFER_LITE_PUBLISH_ROOT}/cxx/lib/*.a)
            endif()
    else()
        if (IOS OR (ARM_TARGET_OS STREQUAL "armlinux"))
            add_custom_target(tiny_publish_lib ${TARGET}
                    COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/lib"
                    COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/include"
                    COMMAND cp "${CMAKE_SOURCE_DIR}/lite/api/paddle_*.h" "${INFER_LITE_PUBLISH_ROOT}/include"
                    COMMAND cp "${CMAKE_BINARY_DIR}/libpaddle_api_light_bundled.a" "${INFER_LITE_PUBLISH_ROOT}/lib"
                    )
            add_dependencies(tiny_publish_lib bundle_light_api)
            add_dependencies(publish_inference tiny_publish_lib)
        endif()
    endif()


    if (LITE_WITH_JAVA)
        # add java lib
        add_custom_target(publish_inference_java_lib ${TARGET}
            COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/java/so"
            COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/java/jar"
            COMMAND cp "${CMAKE_BINARY_DIR}/lite/api/android/jni/native/libpaddle_lite_jni.so" "${INFER_LITE_PUBLISH_ROOT}/java/so"
            COMMAND cp "${CMAKE_BINARY_DIR}/lite/api/android/jni/PaddlePredictor.jar" "${INFER_LITE_PUBLISH_ROOT}/java/jar"
            COMMAND cp -r "${CMAKE_SOURCE_DIR}/lite/api/android/jni/src" "${INFER_LITE_PUBLISH_ROOT}/java"
        )
        add_dependencies(publish_inference_java_lib paddle_lite_jni PaddlePredictor)
        add_dependencies(publish_inference publish_inference_java_lib)
        add_custom_command(TARGET publish_inference_java_lib POST_BUILD
                                   COMMAND ${CMAKE_STRIP} "-s" ${INFER_LITE_PUBLISH_ROOT}/java/so/libpaddle_lite_jni.so)
    endif()

    if ((ARM_TARGET_OS STREQUAL "android") AND (NOT LITE_WITH_OPENCL) AND
            ((ARM_TARGET_ARCH_ABI STREQUAL armv7) OR (ARM_TARGET_ARCH_ABI STREQUAL armv8)))
        if (NOT LITE_ON_TINY_PUBLISH)
            # copy
            add_custom_target(publish_inference_android_cxx_demos ${TARGET}
                COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/third_party"
                COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/demo/cxx"
                COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/demo/cxx/include"
                COMMAND cp -r "${CMAKE_BINARY_DIR}/third_party/install/gflags" "${INFER_LITE_PUBLISH_ROOT}/third_party"
                COMMAND cp "${CMAKE_SOURCE_DIR}/lite/demo/cxx/Makefile.def" "${INFER_LITE_PUBLISH_ROOT}/demo/cxx"
                COMMAND cp "${CMAKE_SOURCE_DIR}/lite/demo/cxx/README.md" "${INFER_LITE_PUBLISH_ROOT}/demo/cxx"
                COMMAND cp -r "${CMAKE_SOURCE_DIR}/lite/demo/cxx/mobile_full" "${INFER_LITE_PUBLISH_ROOT}/demo/cxx"
                COMMAND cp "${CMAKE_SOURCE_DIR}/lite/demo/cxx/makefiles/mobile_full/Makefile.${ARM_TARGET_OS}.${ARM_TARGET_ARCH_ABI}" "${INFER_LITE_PUBLISH_ROOT}/demo/cxx/mobile_full/Makefile"
                COMMAND cp -r "${CMAKE_SOURCE_DIR}/lite/demo/cxx/mobile_light" "${INFER_LITE_PUBLISH_ROOT}/demo/cxx"
                COMMAND cp "${CMAKE_SOURCE_DIR}/lite/demo/cxx/makefiles/mobile_light/Makefile.${ARM_TARGET_OS}.${ARM_TARGET_ARCH_ABI}" "${INFER_LITE_PUBLISH_ROOT}/demo/cxx/mobile_light/Makefile"
            )
            add_dependencies(publish_inference_android_cxx_demos logging gflags)
            add_dependencies(publish_inference_cxx_lib publish_inference_android_cxx_demos)
        endif()

        if (LITE_WITH_JAVA)
            # copy java mobile_light demo/lib
            add_custom_target(publish_inference_android_java_demo ${TARGET}
                    COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/demo/java"
                    COMMAND cp -r "${CMAKE_SOURCE_DIR}/lite/demo/java/android" "${INFER_LITE_PUBLISH_ROOT}/demo/java"
                    COMMAND cp "${CMAKE_SOURCE_DIR}/lite/demo/java/README.md" "${INFER_LITE_PUBLISH_ROOT}/demo/java"
                    COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/demo/java/android/PaddlePredictor/app/libs"
                    COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/demo/java/android/PaddlePredictor/app/src/main/jniLibs/arm7"
                    COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/demo/java/android/PaddlePredictor/app/src/main/jniLibs/arm8"
                    COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/demo/java/android/PaddlePredictor/app/src/main/jniLibs/arm64-v8a"
                    COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/demo/java/android/PaddlePredictor/app/src/main/jniLibs/armeabi-v7a"
                    COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/demo/java/android/PaddlePredictor/app/src/main/jniLibs/x86"
            )
            add_dependencies(publish_inference_java_lib publish_inference_android_java_demo)
        endif()
    endif()

    if (LITE_WITH_OPENCL)
        add_custom_target(publish_inference_opencl ${TARGET}
            COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/opencl"
            COMMAND cp -r "${CMAKE_SOURCE_DIR}/lite/backends/opencl/cl_kernel" "${INFER_LITE_PUBLISH_ROOT}/opencl"
        )
        add_dependencies(publish_inference_cxx_lib publish_inference_opencl)
    endif()
endif()
