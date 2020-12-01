include(lite)

message(STATUS "LIGHT_FRAMEWORK:\t${LITE_WITH_LIGHT_WEIGHT_FRAMEWORK}")
message(STATUS "LITE_WITH_CUDA:\t${LITE_WITH_CUDA}")
message(STATUS "LITE_WITH_X86:\t${LITE_WITH_X86}")
message(STATUS "LITE_WITH_ARM:\t${LITE_WITH_ARM}")
message(STATUS "LITE_WITH_OPENCL:\t${LITE_WITH_OPENCL}")
message(STATUS "LITE_WITH_NPU:\t${LITE_WITH_NPU}")
message(STATUS "LITE_WITH_RKNPU:\t${LITE_WITH_RKNPU}")
message(STATUS "LITE_WITH_XPU:\t${LITE_WITH_XPU}")
message(STATUS "LITE_WITH_APU:\t${LITE_WITH_APU}")
message(STATUS "LITE_WITH_XTCL:\t${LITE_WITH_XTCL}")
message(STATUS "LITE_WITH_FPGA:\t${LITE_WITH_FPGA}")
message(STATUS "LITE_WITH_MLU:\t${LITE_WITH_MLU}")
message(STATUS "LITE_WITH_HUAWEI_ASCEND_NPU:\t${LITE_WITH_HUAWEI_ASCEND_NPU}")
message(STATUS "LITE_WITH_BM:\t${LITE_WITH_BM}")
message(STATUS "LITE_WITH_IMAGINATION_NNA:\t${LITE_WITH_IMAGINATION_NNA}")
message(STATUS "LITE_WITH_PROFILE:\t${LITE_WITH_PROFILE}")
message(STATUS "LITE_WITH_CV:\t${LITE_WITH_CV}")

set(LITE_MODEL_DIR "${THIRD_PARTY_PATH}/install")
set(LITE_ON_MOBILE ${LITE_WITH_LIGHT_WEIGHT_FRAMEWORK})

add_subdirectory(backends)
add_subdirectory(utils)
add_subdirectory(operators)
add_subdirectory(kernels)
add_subdirectory(core)
add_subdirectory(model_parser)
add_subdirectory(fluid)
add_subdirectory(api)

if (NOT LITE_ON_TINY_PUBLISH)
    add_subdirectory(tests)
    add_subdirectory(tools)
endif()
if (LITE_WITH_LIGHT_WEIGHT_FRAMEWORK AND NOT LITE_ON_TINY_PUBLISH)
  add_subdirectory(gen_code)
endif()

if (WITH_TESTING)
    set(LITE_URL_FOR_UNITTESTS "http://paddle-inference-dist.bj.bcebos.com/PaddleLite/models_and_data_for_unittests")
    set(LITE_BAIDU_XPU_INTERNAL_URL_FOR_UNITTESTS $ENV{LITE_BAIDU_XPU_INTERNAL_URL_FOR_UNITTESTS})
    # models
    lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL} "lite_naive_model.tar.gz")
    lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL} "mobilenet_v1.tar.gz")
    lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL} "mobilenet_v2_relu.tar.gz")
    lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL} "inception_v4_simple.tar.gz")
    if(LITE_WITH_LIGHT_WEIGHT_FRAMEWORK)
	    lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL} "mobilenet_v1_int16.tar.gz")
        lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL} "resnet50.tar.gz")
        lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL} "MobileNetV1_quant.tar.gz")
        lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL} "transformer_with_mask_fp32.tar.gz")
        lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL_FOR_UNITTESTS} "squeezenet.tar.gz")
        lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL_FOR_UNITTESTS} "inception_v4.tar.gz")
        lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL_FOR_UNITTESTS} "mobilenet_v3_small_x1_0.tar.gz")
        lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL_FOR_UNITTESTS} "mobilenet_v3_large_x1_0.tar.gz")
        lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL_FOR_UNITTESTS} "mobilenet_v1_int8_for_mediatek_apu.tar.gz")
        lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL_FOR_UNITTESTS} "mobilenet_v1_int8_for_rockchip_npu.tar.gz")
        lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL_FOR_UNITTESTS} "mobilenet_v1_int8_for_imagination_nna.tar.gz")
        lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL_FOR_UNITTESTS} "fast_rcnn_fluid184.tar.gz")
    else()
        lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL} "GoogleNet_inference.tar.gz")
        lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL} "step_rnn.tar.gz")
        lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL_FOR_UNITTESTS} "resnet50.tar.gz")
        lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL_FOR_UNITTESTS} "bert.tar.gz")
        lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL_FOR_UNITTESTS} "ernie.tar.gz")
        lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL_FOR_UNITTESTS} "GoogLeNet.tar.gz")
        lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL_FOR_UNITTESTS} "VGG19.tar.gz")
        if (NOT "${LITE_BAIDU_XPU_INTERNAL_URL_FOR_UNITTESTS}" STREQUAL "")
            lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_BAIDU_XPU_INTERNAL_URL_FOR_UNITTESTS} "mmdnn_model.tar.gz")
            lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_BAIDU_XPU_INTERNAL_URL_FOR_UNITTESTS} "content_dnn_model.tar.gz")
            lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_BAIDU_XPU_INTERNAL_URL_FOR_UNITTESTS} "resnet50_model.tar.gz")
            lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_BAIDU_XPU_INTERNAL_URL_FOR_UNITTESTS} "sfa_model.tar.gz")
        endif()
    endif()
    # data
    lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL_FOR_UNITTESTS} "ILSVRC2012_small.tar.gz")
    lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL_FOR_UNITTESTS} "bert_data.tar.gz")
    if (NOT "${LITE_BAIDU_XPU_INTERNAL_URL_FOR_UNITTESTS}" STREQUAL "")
        lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_BAIDU_XPU_INTERNAL_URL_FOR_UNITTESTS} "mmdnn_data.tar.gz")
        lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_BAIDU_XPU_INTERNAL_URL_FOR_UNITTESTS} "content_dnn_data.tar.gz")
        lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_BAIDU_XPU_INTERNAL_URL_FOR_UNITTESTS} "sfa_data.tar.gz")
    endif()
endif()

# ----------------------------- PUBLISH -----------------------------
# The final target for publish lite lib
add_custom_target(publish_inference)
if (LITE_WITH_LIGHT_WEIGHT_FRAMEWORK AND LITE_WITH_ARM)
    # for publish
    set(INFER_LITE_PUBLISH_ROOT "${CMAKE_BINARY_DIR}/inference_lite_lib.${ARM_TARGET_OS}.${ARM_TARGET_ARCH_ABI}")
    if (LITE_WITH_OPENCL)
        set(INFER_LITE_PUBLISH_ROOT "${INFER_LITE_PUBLISH_ROOT}.opencl")
    endif(LITE_WITH_OPENCL)
    if (LITE_WITH_NPU)
        set(INFER_LITE_PUBLISH_ROOT "${INFER_LITE_PUBLISH_ROOT}.npu")
    endif(LITE_WITH_NPU)
    if (LITE_WITH_XPU)
        set(INFER_LITE_PUBLISH_ROOT "${INFER_LITE_PUBLISH_ROOT}.xpu")
    endif(LITE_WITH_XPU)
    if (LITE_WITH_APU)
        set(INFER_LITE_PUBLISH_ROOT "${INFER_LITE_PUBLISH_ROOT}.apu")
    endif(LITE_WITH_APU)
    if (LITE_WITH_FPGA)
        set(INFER_LITE_PUBLISH_ROOT "${INFER_LITE_PUBLISH_ROOT}.fpga")
    endif(LITE_WITH_FPGA)
    if (LITE_WITH_BM)
        set(INFER_LITE_PUBLISH_ROOT "${INFER_LITE_PUBLISH_ROOT}.bm")
    endif(LITE_WITH_BM)
    if (LITE_WITH_RKNPU)
        set(INFER_LITE_PUBLISH_ROOT "${INFER_LITE_PUBLISH_ROOT}.rknpu")
    endif(LITE_WITH_RKNPU)
    if (LITE_WITH_IMAGINATION_NNA)
        set(INFER_LITE_PUBLISH_ROOT "${INFER_LITE_PUBLISH_ROOT}.nna")
    endif(LITE_WITH_IMAGINATION_NNA)
else()
    set(INFER_LITE_PUBLISH_ROOT "${CMAKE_BINARY_DIR}/inference_lite_lib")
endif()
message(STATUS "publish inference lib to ${INFER_LITE_PUBLISH_ROOT}")

# add python lib
if (LITE_WITH_PYTHON)
    if(WIN32)   
        set(LITE_CORE "${CMAKE_BINARY_DIR}/lite/api/python/pybind/${CMAKE_BUILD_TYPE}/lite.pyd")
        set(LITE_CORE_DEPS ${LITE_CORE})
        add_custom_command(OUTPUT   ${LITE_CORE}
            COMMAND cmake -E copy $<TARGET_FILE:lite_pybind> ${LITE_CORE}
            DEPENDS lite_pybind)
        add_custom_target(copy_lite_pybind ALL DEPENDS ${LITE_CORE_DEPS})
        
        add_custom_target(publish_inference_python_lib ${TARGET}
            COMMAND ${CMAKE_COMMAND} -E make_directory "${INFER_LITE_PUBLISH_ROOT}/python/lib"
            COMMAND ${CMAKE_COMMAND} -E make_directory "${INFER_LITE_PUBLISH_ROOT}/python/install/libs"
            COMMAND ${CMAKE_COMMAND} -E make_directory "${INFER_LITE_PUBLISH_ROOT}/python/install/lite"
            COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_BINARY_DIR}/lite/api/python/setup.py" "${INFER_LITE_PUBLISH_ROOT}/python/install/"
            COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_SOURCE_DIR}/lite/api/python/__init__.py" "${INFER_LITE_PUBLISH_ROOT}/python/install/lite"
            COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_BINARY_DIR}/lite/api/python/pybind/${CMAKE_BUILD_TYPE}/lite.pyd" "${INFER_LITE_PUBLISH_ROOT}/python/install/lite/lite.pyd"
            COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_BINARY_DIR}/lite/api/python/pybind/${CMAKE_BUILD_TYPE}/lite.pyd" "${INFER_LITE_PUBLISH_ROOT}/python/lib/lite.pyd"
            DEPENDS copy_lite_pybind
            )
            
        add_custom_target(publish_inference_python_installer ${TARGET}
            COMMAND ${PYTHON_EXECUTABLE} setup.py bdist_wheel
            WORKING_DIRECTORY ${INFER_LITE_PUBLISH_ROOT}/python/install/
            DEPENDS  publish_inference_python_lib)
        add_custom_target(publish_inference_python_light_demo ${TARGET}
            COMMAND ${CMAKE_COMMAND} -E make_directory "${INFER_LITE_PUBLISH_ROOT}/demo/python"
            COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_SOURCE_DIR}/lite/demo/python/mobilenetv1_light_api.py" "${INFER_LITE_PUBLISH_ROOT}/demo/python/"
            COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_SOURCE_DIR}/lite/demo/python/mobilenetv1_full_api.py" "${INFER_LITE_PUBLISH_ROOT}/demo/python/"
            )
        add_dependencies(publish_inference publish_inference_python_lib)
        add_dependencies(publish_inference publish_inference_python_installer)
        add_dependencies(publish_inference publish_inference_python_light_demo)
    else()
    if(APPLE)
        add_custom_target(publish_inference_python_lib ${TARGET}
                COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/python/lib"
                COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/python/install/libs"
                COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/python/install/lite"
                COMMAND cp "${CMAKE_BINARY_DIR}/lite/api/python/setup.py" "${INFER_LITE_PUBLISH_ROOT}/python/install/"
                COMMAND cp "${CMAKE_SOURCE_DIR}/lite/api/python/__init__.py" "${INFER_LITE_PUBLISH_ROOT}/python/install/lite"
                COMMAND cp "${CMAKE_BINARY_DIR}/lite/api/python/pybind/liblite_pybind.dylib" "${INFER_LITE_PUBLISH_ROOT}/python/install/lite/lite.so"
                COMMAND cp "${CMAKE_BINARY_DIR}/lite/api/python/pybind/liblite_pybind.dylib" "${INFER_LITE_PUBLISH_ROOT}/python/lib/lite.so")
    else()
        add_custom_target(publish_inference_python_lib ${TARGET}
                COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/python/lib"
                COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/python/install/libs"
                COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/python/install/lite"
                COMMAND cp "${CMAKE_BINARY_DIR}/lite/api/python/setup.py" "${INFER_LITE_PUBLISH_ROOT}/python/install/"
                COMMAND cp "${CMAKE_SOURCE_DIR}/lite/api/python/__init__.py" "${INFER_LITE_PUBLISH_ROOT}/python/install/lite"
                COMMAND cp "${CMAKE_BINARY_DIR}/lite/api/python/pybind/liblite_pybind.so" "${INFER_LITE_PUBLISH_ROOT}/python/install/lite/lite.so"
                COMMAND cp "${CMAKE_BINARY_DIR}/lite/api/python/pybind/liblite_pybind.so" "${INFER_LITE_PUBLISH_ROOT}/python/lib/lite.so")
    endif()
    add_custom_target(publish_inference_python_installer ${TARGET}
        COMMAND ${PYTHON_EXECUTABLE} setup.py bdist_wheel
        WORKING_DIRECTORY ${INFER_LITE_PUBLISH_ROOT}/python/install/
        DEPENDS publish_inference_python_lib)
    add_custom_target(publish_inference_python_light_demo ${TARGET}
    	COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/demo/python"
    	COMMAND cp "${CMAKE_SOURCE_DIR}/lite/demo/python/mobilenetv1_light_api.py" "${INFER_LITE_PUBLISH_ROOT}/demo/python/")
    if (NOT LITE_ON_TINY_PUBLISH)
        add_custom_target(publish_inference_python_full_demo ${TARGET}
    	    COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/demo/python"
    	    COMMAND cp "${CMAKE_SOURCE_DIR}/lite/demo/python/mobilenetv1_full_api.py" "${INFER_LITE_PUBLISH_ROOT}/demo/python/")
        add_dependencies(publish_inference publish_inference_python_full_demo)
    endif()
    add_dependencies(publish_inference_python_lib lite_pybind)
    add_dependencies(publish_inference publish_inference_python_lib)
    add_dependencies(publish_inference publish_inference_python_installer)
    add_dependencies(publish_inference publish_inference_python_light_demo)
    endif(WIN32)
endif()

if (LITE_WITH_CUDA OR LITE_WITH_X86)
    if(APPLE)
        add_custom_target(publish_inference_cxx_lib ${TARGET}
            COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/cxx/lib"
            COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/bin"
            COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/cxx/include"
            COMMAND cp "${CMAKE_SOURCE_DIR}/lite/api/paddle_*.h" "${INFER_LITE_PUBLISH_ROOT}/cxx/include"
            COMMAND cp "${CMAKE_BINARY_DIR}/lite/api/*.dylib" "${INFER_LITE_PUBLISH_ROOT}/cxx/lib"
            )
        add_custom_target(publish_inference_third_party ${TARGET}
                COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/third_party"
                COMMAND cp -r "${CMAKE_BINARY_DIR}/third_party/install/*" "${INFER_LITE_PUBLISH_ROOT}/third_party")
        add_dependencies(publish_inference_cxx_lib paddle_full_api_shared)
        add_dependencies(publish_inference_cxx_lib paddle_light_api_shared)
        add_dependencies(publish_inference publish_inference_cxx_lib)
        add_dependencies(publish_inference publish_inference_third_party)
    elseif(NOT WIN32)
        add_custom_target(publish_inference_cxx_lib ${TARGET}
            COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/cxx/lib"
            COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/bin"
            COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/cxx/include"
            COMMAND cp "${CMAKE_SOURCE_DIR}/lite/api/paddle_*.h" "${INFER_LITE_PUBLISH_ROOT}/cxx/include"
            COMMAND cp "${CMAKE_BINARY_DIR}/libpaddle_api_full_bundled.a" "${INFER_LITE_PUBLISH_ROOT}/cxx/lib"
            COMMAND cp "${CMAKE_BINARY_DIR}/libpaddle_api_light_bundled.a" "${INFER_LITE_PUBLISH_ROOT}/cxx/lib"
            COMMAND cp "${CMAKE_BINARY_DIR}/lite/api/*.so" "${INFER_LITE_PUBLISH_ROOT}/cxx/lib"
            )
        if (LITE_WITH_CUDA)
            add_custom_target(publish_inference_third_party ${TARGET}
                    COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/third_party"
                    COMMAND cp -r "${CMAKE_BINARY_DIR}/third_party/install/*" "${INFER_LITE_PUBLISH_ROOT}/third_party")
            add_dependencies(publish_inference publish_inference_third_party)
        endif()
        add_dependencies(publish_inference_cxx_lib bundle_full_api)
        add_dependencies(publish_inference_cxx_lib bundle_light_api)
        add_dependencies(publish_inference_cxx_lib paddle_full_api_shared)
        add_dependencies(publish_inference_cxx_lib paddle_light_api_shared)
        add_dependencies(publish_inference publish_inference_cxx_lib)
    endif()
endif()

if (LITE_WITH_X86)
  if(WIN32)
        add_custom_target(publish_inference_x86_cxx_lib ${TARGET}
            COMMAND ${CMAKE_COMMAND} -E make_directory "${INFER_LITE_PUBLISH_ROOT}/cxx/lib"
            COMMAND ${CMAKE_COMMAND} -E make_directory "${INFER_LITE_PUBLISH_ROOT}/bin"
            COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_BINARY_DIR}/lite/api//${CMAKE_BUILD_TYPE}/test_model_bin.exe" "${INFER_LITE_PUBLISH_ROOT}/bin"
            COMMAND ${CMAKE_COMMAND} -E make_directory "${INFER_LITE_PUBLISH_ROOT}/cxx/include"
            COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_SOURCE_DIR}/lite/api/paddle_api.h" "${INFER_LITE_PUBLISH_ROOT}/cxx/include"
            COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_SOURCE_DIR}/lite/api/paddle_place.h" "${INFER_LITE_PUBLISH_ROOT}/cxx/include"
            COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_SOURCE_DIR}/lite/api/paddle_use_kernels.h" "${INFER_LITE_PUBLISH_ROOT}/cxx/include"
            COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_SOURCE_DIR}/lite/api/paddle_use_ops.h" "${INFER_LITE_PUBLISH_ROOT}/cxx/include"
            COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_SOURCE_DIR}/lite/api/paddle_use_passes.h" "${INFER_LITE_PUBLISH_ROOT}/cxx/include"
            COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_SOURCE_DIR}/lite/api/paddle_lite_factory_helper.h" "${INFER_LITE_PUBLISH_ROOT}/cxx/include"
            COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_BINARY_DIR}/lite/api/${CMAKE_BUILD_TYPE}/libpaddle_api_full_bundled.lib" "${INFER_LITE_PUBLISH_ROOT}/cxx/lib"
            COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_BINARY_DIR}/lite/api/${CMAKE_BUILD_TYPE}/libpaddle_api_light_bundled.lib" "${INFER_LITE_PUBLISH_ROOT}/cxx/lib"
        )
       
        add_dependencies(publish_inference_x86_cxx_lib test_model_bin)
        add_dependencies(publish_inference_x86_cxx_lib bundle_full_api)
        add_dependencies(publish_inference_x86_cxx_lib bundle_light_api)
        add_dependencies(publish_inference publish_inference_x86_cxx_lib)

        configure_file(${CMAKE_SOURCE_DIR}/lite/demo/cxx/x86_mobilenetv1_light_demo/CMakeLists.txt.in ${CMAKE_SOURCE_DIR}/lite/demo/cxx/x86_mobilenetv1_light_demo/CMakeLists.txt @ONLY)
        configure_file(${CMAKE_SOURCE_DIR}/lite/demo/cxx/x86_mobilenetv1_full_demo/CMakeLists.txt.in ${CMAKE_SOURCE_DIR}/lite/demo/cxx/x86_mobilenetv1_full_demo/CMakeLists.txt @ONLY)
        configure_file(${CMAKE_SOURCE_DIR}/lite/demo/cxx/x86_mobilenetv1_light_demo/build.bat.in ${CMAKE_SOURCE_DIR}/lite/demo/cxx/x86_mobilenetv1_light_demo/build.bat @ONLY)
        configure_file(${CMAKE_SOURCE_DIR}/lite/demo/cxx/x86_mobilenetv1_full_demo/build.bat.in ${CMAKE_SOURCE_DIR}/lite/demo/cxx/x86_mobilenetv1_full_demo/build.bat @ONLY)

        add_custom_target(publish_inference_x86_cxx_demos ${TARGET}
            COMMAND ${CMAKE_COMMAND} -E make_directory "${INFER_LITE_PUBLISH_ROOT}/third_party/mklml"
            COMMAND ${CMAKE_COMMAND} -E copy_directory "${CMAKE_BINARY_DIR}/third_party/install/mklml" "${INFER_LITE_PUBLISH_ROOT}/third_party/mklml"
            COMMAND ${CMAKE_COMMAND} -E make_directory "${INFER_LITE_PUBLISH_ROOT}/demo/cxx"
            COMMAND ${CMAKE_COMMAND} -E copy_directory "${CMAKE_SOURCE_DIR}/lite/demo/cxx/x86_mobilenetv1_light_demo" "${INFER_LITE_PUBLISH_ROOT}/demo/cxx/mobilenetv1_light"
            COMMAND ${CMAKE_COMMAND} -E copy_directory "${CMAKE_SOURCE_DIR}/lite/demo/cxx/x86_mobilenetv1_full_demo" "${INFER_LITE_PUBLISH_ROOT}/demo/cxx/mobilenetv1_full"
            COMMAND ${CMAKE_COMMAND} -E remove "${INFER_LITE_PUBLISH_ROOT}/demo/cxx/mobilenetv1_light/CMakeLists.txt.in"
            COMMAND ${CMAKE_COMMAND} -E remove "${INFER_LITE_PUBLISH_ROOT}/demo/cxx/mobilenetv1_full/CMakeLists.txt.in"
            COMMAND ${CMAKE_COMMAND} -E remove "${INFER_LITE_PUBLISH_ROOT}/demo/cxx/mobilenetv1_light/build.bat.in"
            COMMAND ${CMAKE_COMMAND} -E remove "${INFER_LITE_PUBLISH_ROOT}/demo/cxx/mobilenetv1_full/build.bat.in"
            COMMAND ${CMAKE_COMMAND} -E remove "${CMAKE_SOURCE_DIR}/lite/demo/cxx/x86_mobilenetv1_light_demo/CMakeLists.txt"
            COMMAND ${CMAKE_COMMAND} -E remove "${CMAKE_SOURCE_DIR}/lite/demo/cxx/x86_mobilenetv1_full_demo/CMakeLists.txt"
        )
        add_dependencies(publish_inference_x86_cxx_lib publish_inference_x86_cxx_demos)
        add_dependencies(publish_inference_x86_cxx_demos paddle_api_full_bundled eigen3)

  else()
    add_custom_target(publish_inference_x86_cxx_lib ${TARGET}
            COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/bin"
            COMMAND cp "${CMAKE_BINARY_DIR}/lite/api/test_model_bin" "${INFER_LITE_PUBLISH_ROOT}/bin"
            )
    add_dependencies(publish_inference_x86_cxx_lib test_model_bin)

    configure_file(${CMAKE_SOURCE_DIR}/lite/demo/cxx/x86_mobilenetv1_light_demo/CMakeLists.txt.in ${CMAKE_SOURCE_DIR}/lite/demo/cxx/x86_mobilenetv1_light_demo/CMakeLists.txt @ONLY)
    configure_file(${CMAKE_SOURCE_DIR}/lite/demo/cxx/x86_mobilenetv1_full_demo/CMakeLists.txt.in ${CMAKE_SOURCE_DIR}/lite/demo/cxx/x86_mobilenetv1_full_demo/CMakeLists.txt @ONLY)   
    add_custom_target(publish_inference_x86_cxx_demos ${TARGET}
           COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/demo/cxx"
           COMMAND cp -r "${CMAKE_SOURCE_DIR}/lite/demo/cxx/x86_mobilenetv1_light_demo" "${INFER_LITE_PUBLISH_ROOT}/demo/cxx/mobilenetv1_light"
           COMMAND cp -r "${CMAKE_SOURCE_DIR}/lite/demo/cxx/x86_mobilenetv1_full_demo" "${INFER_LITE_PUBLISH_ROOT}/demo/cxx/mobilenetv1_full"
           COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/third_party"
           COMMAND rm -rf "${INFER_LITE_PUBLISH_ROOT}/demo/cxx/mobilenetv1_light/*.in"
           COMMAND rm -rf "${INFER_LITE_PUBLISH_ROOT}/demo/cxx/mobilenetv1_full/*.in"
           COMMAND rm -rf "${CMAKE_SOURCE_DIR}/lite/demo/cxx/x86_mobilenetv1_light_demo/CMakeLists.txt"
           COMMAND rm -rf "${CMAKE_SOURCE_DIR}/lite/demo/cxx/x86_mobilenetv1_full_demo/CMakeLists.txt"
       )
    if(WITH_MKL)
        add_custom_command(TARGET publish_inference_x86_cxx_demos POST_BUILD
            COMMAND cp -r "${CMAKE_BINARY_DIR}/third_party/install/mklml" "${INFER_LITE_PUBLISH_ROOT}/third_party/")
    endif()
    add_dependencies(publish_inference_x86_cxx_lib publish_inference_x86_cxx_demos)
    add_dependencies(publish_inference_x86_cxx_demos paddle_full_api_shared eigen3)
    add_dependencies(publish_inference publish_inference_x86_cxx_lib)
    add_dependencies(publish_inference publish_inference_x86_cxx_demos)
  endif()
endif()

if(LITE_WITH_CUDA)
    add_custom_target(publish_inference_cuda_cxx_demos ${TARGET}
           COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/demo/cxx"
           COMMAND cp -r "${CMAKE_SOURCE_DIR}/lite/demo/cxx/cuda_demo/*" "${INFER_LITE_PUBLISH_ROOT}/demo/cxx"
           )
    add_dependencies(publish_inference_cuda_cxx_demos paddle_full_api_shared)
    add_dependencies(publish_inference publish_inference_cuda_cxx_demos)
endif(LITE_WITH_CUDA)

if (LITE_WITH_LIGHT_WEIGHT_FRAMEWORK AND LITE_WITH_ARM)
    if (NOT LITE_ON_TINY_PUBLISH)
        # add cxx lib
        add_custom_target(publish_inference_cxx_lib ${TARGET}
                COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/cxx/lib"
                COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/bin"
                COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/cxx/include"
                COMMAND cp "${CMAKE_SOURCE_DIR}/lite/api/paddle_*.h" "${INFER_LITE_PUBLISH_ROOT}/cxx/include"
                COMMAND cp "${CMAKE_BINARY_DIR}/libpaddle_api_full_bundled.a" "${INFER_LITE_PUBLISH_ROOT}/cxx/lib"
                COMMAND cp "${CMAKE_BINARY_DIR}/libpaddle_api_light_bundled.a" "${INFER_LITE_PUBLISH_ROOT}/cxx/lib"
                COMMAND cp "${CMAKE_BINARY_DIR}/lite/gen_code/paddle_code_generator" "${INFER_LITE_PUBLISH_ROOT}/bin"
                COMMAND cp "${CMAKE_BINARY_DIR}/lite/api/test_model_bin" "${INFER_LITE_PUBLISH_ROOT}/bin"
                COMMAND cp "${CMAKE_SOURCE_DIR}/lite/utils/cv/paddle_*.h" "${INFER_LITE_PUBLISH_ROOT}/cxx/include"
                )
            if(NOT IOS)
                add_dependencies(publish_inference_cxx_lib paddle_code_generator)
                add_dependencies(publish_inference_cxx_lib bundle_full_api)
                add_dependencies(publish_inference_cxx_lib bundle_light_api)
                add_dependencies(publish_inference_cxx_lib test_model_bin)
                add_dependencies(publish_inference_cxx_lib benchmark_bin)
                if (ARM_TARGET_OS STREQUAL "android" OR ARM_TARGET_OS STREQUAL "armlinux")
                    add_dependencies(publish_inference_cxx_lib paddle_full_api_shared)
                    add_dependencies(publish_inference paddle_light_api_shared)
                    add_custom_command(TARGET publish_inference_cxx_lib
                          COMMAND cp ${CMAKE_BINARY_DIR}/lite/api/*.so ${INFER_LITE_PUBLISH_ROOT}/cxx/lib
                          COMMAND cp "${CMAKE_BINARY_DIR}/lite/api/benchmark_bin" "${INFER_LITE_PUBLISH_ROOT}/bin"
                          )
                endif()
                add_dependencies(publish_inference publish_inference_cxx_lib)
                if(NOT "${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
                    add_custom_command(TARGET publish_inference_cxx_lib POST_BUILD
                            COMMAND ${CMAKE_STRIP} "--strip-debug" ${INFER_LITE_PUBLISH_ROOT}/cxx/lib/*.a
                            COMMAND ${CMAKE_STRIP} "--strip-debug" ${INFER_LITE_PUBLISH_ROOT}/cxx/lib/*.so)
                endif()
            endif()
    else()
        if (IOS)
            add_custom_target(tiny_publish_lib ${TARGET}
                    COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/lib"
                    COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/include"
                    COMMAND cp "${CMAKE_SOURCE_DIR}/lite/api/paddle_*.h" "${INFER_LITE_PUBLISH_ROOT}/include"
                    COMMAND cp "${CMAKE_BINARY_DIR}/libpaddle_api_light_bundled.a" "${INFER_LITE_PUBLISH_ROOT}/lib"
                    COMMAND cp "${CMAKE_SOURCE_DIR}/lite/utils/cv/paddle_*.h" "${INFER_LITE_PUBLISH_ROOT}/include"
                    )
            add_dependencies(tiny_publish_lib bundle_light_api)
            add_dependencies(publish_inference tiny_publish_lib)
        else()
            if ((ARM_TARGET_OS STREQUAL "android") OR (ARM_TARGET_OS STREQUAL "armlinux"))
                add_custom_target(tiny_publish_cxx_lib ${TARGET}
                    COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/cxx"
                    COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/cxx/include"
                    COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/cxx/lib"
                    COMMAND cp "${CMAKE_SOURCE_DIR}/lite/api/paddle_*.h" "${INFER_LITE_PUBLISH_ROOT}/cxx/include"
                    COMMAND cp "${CMAKE_BINARY_DIR}/libpaddle_api_light_bundled.a" "${INFER_LITE_PUBLISH_ROOT}/cxx/lib"
                    COMMAND cp "${CMAKE_BINARY_DIR}/lite/api/libpaddle_light_api_shared.so" "${INFER_LITE_PUBLISH_ROOT}/cxx/lib"
                    COMMAND cp "${CMAKE_SOURCE_DIR}/lite/utils/cv/paddle_*.h" "${INFER_LITE_PUBLISH_ROOT}/cxx/include"
                    )
                add_dependencies(tiny_publish_cxx_lib paddle_light_api_shared)
                add_dependencies(tiny_publish_cxx_lib bundle_light_api)
                add_dependencies(publish_inference tiny_publish_cxx_lib)
                if(NOT "${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
                    add_custom_command(TARGET tiny_publish_cxx_lib POST_BUILD
                                COMMAND ${CMAKE_STRIP} "-s" ${INFER_LITE_PUBLISH_ROOT}/cxx/lib/libpaddle_light_api_shared.so)
                endif()
            endif()
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
        if(NOT "${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
            add_custom_command(TARGET publish_inference_java_lib POST_BUILD
                                       COMMAND ${CMAKE_STRIP} "-s" ${INFER_LITE_PUBLISH_ROOT}/java/so/libpaddle_lite_jni.so)
        endif()
    endif()

    if ((ARM_TARGET_OS STREQUAL "android") AND
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
                COMMAND cp -r "${CMAKE_SOURCE_DIR}/lite/demo/cxx/ssd_detection" "${INFER_LITE_PUBLISH_ROOT}/demo/cxx"
                COMMAND cp "${CMAKE_SOURCE_DIR}/lite/demo/cxx/makefiles/ssd_detection/Makefile.${ARM_TARGET_OS}.${ARM_TARGET_ARCH_ABI}" "${INFER_LITE_PUBLISH_ROOT}/demo/cxx/ssd_detection/Makefile"
                COMMAND cp -r "${CMAKE_SOURCE_DIR}/lite/demo/cxx/yolov3_detection" "${INFER_LITE_PUBLISH_ROOT}/demo/cxx"
                COMMAND cp "${CMAKE_SOURCE_DIR}/lite/demo/cxx/makefiles/yolov3_detection/Makefile.${ARM_TARGET_OS}.${ARM_TARGET_ARCH_ABI}" "${INFER_LITE_PUBLISH_ROOT}/demo/cxx/yolov3_detection/Makefile"
                COMMAND cp -r "${CMAKE_SOURCE_DIR}/lite/demo/cxx/mobile_classify" "${INFER_LITE_PUBLISH_ROOT}/demo/cxx"
                COMMAND cp "${CMAKE_SOURCE_DIR}/lite/demo/cxx/makefiles/mobile_classify/Makefile.${ARM_TARGET_OS}.${ARM_TARGET_ARCH_ABI}" "${INFER_LITE_PUBLISH_ROOT}/demo/cxx/mobile_classify/Makefile"
                COMMAND cp -r "${CMAKE_SOURCE_DIR}/lite/demo/cxx/test_cv" "${INFER_LITE_PUBLISH_ROOT}/demo/cxx"
                COMMAND cp "${CMAKE_SOURCE_DIR}/lite/demo/cxx/makefiles/test_cv/Makefile.${ARM_TARGET_OS}.${ARM_TARGET_ARCH_ABI}" "${INFER_LITE_PUBLISH_ROOT}/demo/cxx/test_cv/Makefile"
                COMMAND cp -r "${CMAKE_SOURCE_DIR}/lite/demo/cxx/mask_detection" "${INFER_LITE_PUBLISH_ROOT}/demo/cxx"
                COMMAND cp "${CMAKE_SOURCE_DIR}/lite/demo/cxx/makefiles/mask_detection/Makefile.${ARM_TARGET_OS}.${ARM_TARGET_ARCH_ABI}" "${INFER_LITE_PUBLISH_ROOT}/demo/cxx/mask_detection/Makefile"
                COMMAND cp -r "${CMAKE_SOURCE_DIR}/lite/demo/cxx/test_libs" "${INFER_LITE_PUBLISH_ROOT}/demo/cxx"
                COMMAND cp "${CMAKE_SOURCE_DIR}/lite/demo/cxx/makefiles/test_libs/Makefile.${ARM_TARGET_OS}.${ARM_TARGET_ARCH_ABI}" "${INFER_LITE_PUBLISH_ROOT}/demo/cxx/test_libs/Makefile"
                COMMAND cp -r "${CMAKE_SOURCE_DIR}/lite/demo/cxx/quant_post_dynamic" "${INFER_LITE_PUBLISH_ROOT}/demo/cxx"
                COMMAND cp "${CMAKE_SOURCE_DIR}/lite/demo/cxx/makefiles/quant_post_dynamic/Makefile.${ARM_TARGET_OS}.${ARM_TARGET_ARCH_ABI}" "${INFER_LITE_PUBLISH_ROOT}/demo/cxx/quant_post_dynamic/Makefile"
            )
            add_dependencies(publish_inference_android_cxx_demos logging gflags)
            add_dependencies(publish_inference_cxx_lib publish_inference_android_cxx_demos)
        else()
            # copy
            add_custom_target(publish_inference_android_cxx_demos ${TARGET}
                COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/demo/cxx"
                COMMAND cp "${CMAKE_SOURCE_DIR}/lite/demo/cxx/Makefile.def" "${INFER_LITE_PUBLISH_ROOT}/demo/cxx"
                COMMAND cp "${CMAKE_SOURCE_DIR}/lite/demo/cxx/README.md" "${INFER_LITE_PUBLISH_ROOT}/demo/cxx"
                COMMAND cp -r "${CMAKE_SOURCE_DIR}/lite/demo/cxx/mobile_light" "${INFER_LITE_PUBLISH_ROOT}/demo/cxx"
                COMMAND cp "${CMAKE_SOURCE_DIR}/lite/demo/cxx/makefiles/mobile_light/Makefile.${ARM_TARGET_OS}.${ARM_TARGET_ARCH_ABI}" "${INFER_LITE_PUBLISH_ROOT}/demo/cxx/mobile_light/Makefile"
                COMMAND cp -r "${CMAKE_SOURCE_DIR}/lite/demo/cxx/ssd_detection" "${INFER_LITE_PUBLISH_ROOT}/demo/cxx"
                COMMAND cp "${CMAKE_SOURCE_DIR}/lite/demo/cxx/makefiles/ssd_detection/Makefile.${ARM_TARGET_OS}.${ARM_TARGET_ARCH_ABI}" "${INFER_LITE_PUBLISH_ROOT}/demo/cxx/ssd_detection/Makefile"
                COMMAND cp -r "${CMAKE_SOURCE_DIR}/lite/demo/cxx/yolov3_detection" "${INFER_LITE_PUBLISH_ROOT}/demo/cxx"
                COMMAND cp "${CMAKE_SOURCE_DIR}/lite/demo/cxx/makefiles/yolov3_detection/Makefile.${ARM_TARGET_OS}.${ARM_TARGET_ARCH_ABI}" "${INFER_LITE_PUBLISH_ROOT}/demo/cxx/yolov3_detection/Makefile"
                COMMAND cp -r "${CMAKE_SOURCE_DIR}/lite/demo/cxx/mobile_classify" "${INFER_LITE_PUBLISH_ROOT}/demo/cxx"
                COMMAND cp "${CMAKE_SOURCE_DIR}/lite/demo/cxx/makefiles/mobile_classify/Makefile.${ARM_TARGET_OS}.${ARM_TARGET_ARCH_ABI}" "${INFER_LITE_PUBLISH_ROOT}/demo/cxx/mobile_classify/Makefile"
                COMMAND cp -r "${CMAKE_SOURCE_DIR}/lite/demo/cxx/test_cv" "${INFER_LITE_PUBLISH_ROOT}/demo/cxx"
                COMMAND cp "${CMAKE_SOURCE_DIR}/lite/demo/cxx/makefiles/test_cv/Makefile.${ARM_TARGET_OS}.${ARM_TARGET_ARCH_ABI}" "${INFER_LITE_PUBLISH_ROOT}/demo/cxx/test_cv/Makefile"
                COMMAND cp -r "${CMAKE_SOURCE_DIR}/lite/demo/cxx/mask_detection" "${INFER_LITE_PUBLISH_ROOT}/demo/cxx"
                COMMAND cp "${CMAKE_SOURCE_DIR}/lite/demo/cxx/makefiles/mask_detection/Makefile.${ARM_TARGET_OS}.${ARM_TARGET_ARCH_ABI}" "${INFER_LITE_PUBLISH_ROOT}/demo/cxx/mask_detection/Makefile"
            )
            add_dependencies(tiny_publish_cxx_lib publish_inference_android_cxx_demos)
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
       if (NOT LITE_ON_TINY_PUBLISH)
        add_dependencies(publish_inference_cxx_lib publish_inference_opencl)
       else()
        add_dependencies(tiny_publish_cxx_lib publish_inference_opencl)
       endif()
    endif()
endif()

if (ARM_TARGET_OS STREQUAL "armlinux")
    if (NOT LITE_ON_TINY_PUBLISH)
	add_custom_target(publish_inference_armlinux_cxx_demos ${TARGET}
            COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/demo/cxx"
            COMMAND cp -r "${CMAKE_SOURCE_DIR}/lite/demo/cxx/armlinux_mobilenetv1_full_demo" "${INFER_LITE_PUBLISH_ROOT}/demo/cxx/mobilenetv1_full"
            COMMAND cp -r "${CMAKE_SOURCE_DIR}/lite/demo/cxx/armlinux_mobilenetv1_light_demo" "${INFER_LITE_PUBLISH_ROOT}/demo/cxx/mobilenetv1_light"
        )
    else()
	add_custom_target(publish_inference_armlinux_cxx_demos ${TARGET}
            COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/demo/cxx"
            COMMAND cp -r "${CMAKE_SOURCE_DIR}/lite/demo/cxx/armlinux_mobilenetv1_light_demo" "${INFER_LITE_PUBLISH_ROOT}/demo/cxx/mobilenetv1_light"
        )
    endif()
    add_dependencies(publish_inference publish_inference_armlinux_cxx_demos)
endif()

if (LITE_WITH_BM)
    add_custom_target(publish_inference_bm_cxx_demos ${TARGET}
        COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/demo/cxx"
        COMMAND cp "${CMAKE_SOURCE_DIR}/lite/demo/cxx/bm_demo/*" "${INFER_LITE_PUBLISH_ROOT}/demo/cxx/"
        COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/cxx"
        COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/cxx/lib"
        COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/cxx/lib/third_party"
        COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/cxx/include"
        COMMAND cp "${CMAKE_BINARY_DIR}/lite/api/libpaddle_full_api_shared.so" "${INFER_LITE_PUBLISH_ROOT}/cxx/lib/"
        COMMAND cp "${BM_SDK_ROOT}/lib/bmcompiler/*" "${INFER_LITE_PUBLISH_ROOT}/cxx/lib/third_party/"
        COMMAND cp "${BM_SDK_ROOT}/lib/bmnn/pcie/*" "${INFER_LITE_PUBLISH_ROOT}/cxx/lib/third_party/"
        COMMAND cp "${CMAKE_SOURCE_DIR}/lite/api/paddle_*.h" "${INFER_LITE_PUBLISH_ROOT}/cxx/include/"
    )
    add_dependencies(publish_inference_bm_cxx_demos paddle_full_api_shared)
    add_dependencies(publish_inference publish_inference_bm_cxx_demos)
endif()
if (LITE_WITH_MLU)
    add_custom_target(publish_inference_mlu_cxx_demos ${TARGET}
        COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/demo/cxx"
        COMMAND cp -r "${CMAKE_SOURCE_DIR}/lite/demo/cxx/mlu_demo" "${INFER_LITE_PUBLISH_ROOT}/demo/cxx"
        COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/demo/cxx/mlu_demo/lib"
        COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/demo/cxx/mlu_demo/include"
        COMMAND cp -r "${CMAKE_BINARY_DIR}/lite/api/libpaddle_full_api_shared.so" "${INFER_LITE_PUBLISH_ROOT}/demo/cxx/mlu_demo/lib"
        COMMAND cp -r "${CMAKE_SOURCE_DIR}/lite/api/paddle_place.h" "${INFER_LITE_PUBLISH_ROOT}/demo/cxx/mlu_demo/include"
        COMMAND cp -r "${CMAKE_SOURCE_DIR}/lite/api/paddle_api.h" "${INFER_LITE_PUBLISH_ROOT}/demo/cxx/mlu_demo/include"
    )
    add_dependencies(publish_inference_mlu_cxx_demos paddle_full_api_shared)
    add_dependencies(publish_inference publish_inference_mlu_cxx_demos)
endif()
