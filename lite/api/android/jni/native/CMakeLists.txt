# Generate paddle_lite_jni.so

if (LITE_ON_TINY_PUBLISH)
    set(CMAKE_CXX_FLAGS_RELEASE "-Os -DNDEBUG")
    set(CMAKE_C_FLAGS_RELEASE "-Os -DNDEBUG")
    set(lib_DEPS light_api paddle_api paddle_api_light)
else()
    set(lib_DEPS light_api cxx_api paddle_api_full paddle_api paddle_api_light)
endif()

include_directories(${JNI_INCLUDE_DIRS} ${_classDir} ${_stubDir})
if (NOT LITE_ON_TINY_PUBLISH)
    lite_cc_library(paddle_lite_jni MODULE 
        SRCS paddle_lite_jni.cc tensor_jni.cc
        DEPS ${lib_DEPS}
        ARM_DEPS ${arm_kernels} NPU_DEPS ${npu_kernels})
    # Unlike static library, module library has to link target to be able to work
    # as a single .so lib.
    target_link_libraries(paddle_lite_jni ${lib_DEPS} ${arm_kernels} ${npu_kernels})
else()
    add_library(paddle_lite_jni SHARED "")
    target_sources(paddle_lite_jni PUBLIC ${__lite_cc_files} paddle_lite_jni.cc tensor_jni.cc)
    add_dependencies(paddle_lite_jni op_list_h kernel_list_h)
endif()

if (APPLE)
    # MacOS only accepts JNI lib ends with .jnilib or .dylib
    set_target_properties(paddle_lite_jni PROPERTIES SUFFIX ".jnilib")
elseif (WIN32)
    # Windows only accepts JNI lib ends with .dll
    set_target_properties(paddle_lite_jni PROPERTIES SUFFIX ".dll")
endif (APPLE)
