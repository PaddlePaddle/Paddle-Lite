# Generate paddle_lite_jni.so

if (LITE_ON_TINY_PUBLISH)
    set(CMAKE_CXX_FLAGS_RELEASE "-Os -DNDEBUG")
    set(CMAKE_C_FLAGS_RELEASE "-Os -DNDEBUG")
    set(lib_DEPS paddle_api_light model_parser utils core kernels ops)
else()
    set(lib_DEPS paddle_api_full model_parser utils core kernels ops)
endif()

include_directories(${JNI_INCLUDE_DIRS} ${_classDir} ${_stubDir})
if (NOT LITE_ON_TINY_PUBLISH)
    lite_cc_library(paddle_lite_jni MODULE 
        SRCS paddle_lite_jni.cc tensor_jni.cc
        DEPS ${lib_DEPS} kernels)
    # Unlike static library, module library has to link target to be able to work
    # as a single .so lib.
    target_link_libraries(paddle_lite_jni ${lib_DEPS})
    if (LITE_WITH_NPU)
        # Strips the symbols of our protobuf functions to fix the conflicts during
        # loading HIAI builder libs (libhiai_ir.so and libhiai_ir_build.so)
        set(LINK_FLAGS "-Wl,--version-script ${PADDLE_SOURCE_DIR}/lite/core/lite.map")
        set_target_properties(paddle_lite_jni PROPERTIES LINK_FLAGS "${LINK_FLAGS}")
    endif()
else()
  # In tiny_pubish mode, java lib paddle_lite_jni is defined in lite/api/CMakeList.txt
  # to accelate its compiling.
endif()

if (APPLE)
    # MacOS only accepts JNI lib ends with .jnilib or .dylib
    set_target_properties(paddle_lite_jni PROPERTIES SUFFIX ".jnilib")
elseif (WIN32)
    # Windows only accepts JNI lib ends with .dll
    set_target_properties(paddle_lite_jni PROPERTIES SUFFIX ".dll")
endif (APPLE)
