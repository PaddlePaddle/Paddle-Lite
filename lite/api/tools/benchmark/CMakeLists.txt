if(ARM_TARGET_OS STREQUAL "android")
    if ("${ARM_TARGET_ARCH_ABI}" STREQUAL "armv8")
        set(ARM_ABI "arm64-v8a")
    elseif("${ARM_TARGET_ARCH_ABI}" STREQUAL "armv7")
        set(ARM_ABI "armeabi-v7a")
    endif()

    # OpenCV + imagenet validation assets are ARM-only in upstream bundles.
    if(NOT ("${ARM_TARGET_ARCH_ABI}" STREQUAL "x86_64" OR "${ARM_TARGET_ARCH_ABI}" STREQUAL "x86"))
    set(OPENCV_VERSION "opencv4.1.0")
    set(OPENCV_PATH "${THIRD_PARTY_PATH}/${OPENCV_VERSION}/${ARM_ABI}")
    message(STATUS "OPENCV_PATH: ${OPENCV_PATH}")
    set(OPENCV_INCLUDE_DIRS "${OPENCV_PATH}/include/")
    set(OPENCV_LIBS ${OPENCV_PATH}/libs/libopencv_imgcodecs.a
                    ${OPENCV_PATH}/libs/libopencv_imgproc.a
                    ${OPENCV_PATH}/libs/libopencv_core.a
                    ${OPENCV_PATH}/3rdparty/libs/libtegra_hal.a
                    ${OPENCV_PATH}/3rdparty/libs/liblibjpeg-turbo.a
                    ${OPENCV_PATH}/3rdparty/libs/liblibwebp.a
                    ${OPENCV_PATH}/3rdparty/libs/liblibpng.a
                    ${OPENCV_PATH}/3rdparty/libs/liblibjasper.a
                    ${OPENCV_PATH}/3rdparty/libs/liblibtiff.a
                    ${OPENCV_PATH}/3rdparty/libs/libIlmImf.a
                    ${OPENCV_PATH}/3rdparty/libs/libtbb.a
                    ${OPENCV_PATH}/3rdparty/libs/libcpufeatures.a)
    set(OPENCV_URL "http://paddle-inference-dist.bj.bcebos.com")
    lite_download_and_uncompress(${THIRD_PARTY_PATH} ${OPENCV_URL} "${OPENCV_VERSION}.tar.gz")

    set(VALIDATION_DATASET_DIR "${THIRD_PARTY_PATH}/validation_dataset")
    set(VALIDATION_DATASET_URL "http://paddle-inference-dist.bj.bcebos.com/AI-Rank/validation_dataset")
    set(ILSVRC2012_1K_CLS_IMGS "ILSVRC2012_1000_cls")
    set(ILSVRC2012_1K_CLS_LABELS "ILSVRC2012_1000_cls_label_list")
    lite_download_and_uncompress(${VALIDATION_DATASET_DIR} ${VALIDATION_DATASET_URL} "${ILSVRC2012_1K_CLS_IMGS}.tar.gz")
    lite_download_and_uncompress(${VALIDATION_DATASET_DIR} ${VALIDATION_DATASET_URL} "${ILSVRC2012_1K_CLS_LABELS}.tar.gz")
    endif()
endif()

FILE(GLOB_RECURSE BENCHMARK_SRC *.cc)
if(NOT(ANDROID AND LITE_WITH_X86 AND LITE_ON_TINY_PUBLISH))
    list(APPEND BENCHMARK_SRC ../opt_base.cc)
endif()
if(NOT(ARM_TARGET_OS STREQUAL "android"))
    FILE(GLOB_RECURSE BENCHMARK_PRECISION_SRC precision_evaluation/*.cc)
    LIST(REMOVE_ITEM BENCHMARK_SRC ${BENCHMARK_PRECISION_SRC})
elseif("${ARM_TARGET_ARCH_ABI}" STREQUAL "x86_64" OR "${ARM_TARGET_ARCH_ABI}" STREQUAL "x86")
    FILE(GLOB_RECURSE BENCHMARK_PRECISION_SRC precision_evaluation/*.cc)
    LIST(REMOVE_ITEM BENCHMARK_SRC ${BENCHMARK_PRECISION_SRC})
endif()
if(APPLE)
    message(STATUS "benchmark monitor is only available on android/linux os.")
    FILE(GLOB_RECURSE BENCHMARK_MONITOR_SRC profile/*.cc)
    LIST(REMOVE_ITEM BENCHMARK_SRC ${BENCHMARK_MONITOR_SRC})
elseif(ANDROID AND ("${ARM_TARGET_ARCH_ABI}" STREQUAL "x86_64" OR "${ARM_TARGET_ARCH_ABI}" STREQUAL "x86"))
    message(STATUS "benchmark monitor is only available on android/linux os.")
    FILE(GLOB_RECURSE BENCHMARK_MONITOR_SRC profile/*.cc)
    LIST(REMOVE_ITEM BENCHMARK_SRC ${BENCHMARK_MONITOR_SRC})
endif()

set(TARGET "benchmark_bin")
if(ANDROID AND LITE_WITH_X86 AND LITE_ON_TINY_PUBLISH)
    # tiny-publish: link prebuilt light shared lib (avoids broken static/third-party relink).
    # glog.cmake sets directory LINK_LIBRARIES(glog gflags); clear so we do not
    # statically duplicate glog/gflags already baked into libpaddle_light_api_shared.so.
    get_property(_benchmark_saved_dir_link_libs DIRECTORY PROPERTY LINK_LIBRARIES)
    set_property(DIRECTORY PROPERTY LINK_LIBRARIES "")
    add_executable(${TARGET} ${BENCHMARK_SRC})
    set_property(DIRECTORY PROPERTY LINK_LIBRARIES ${_benchmark_saved_dir_link_libs})
    add_dependencies(${TARGET} fbs_headers)
    target_compile_definitions(${TARGET} PRIVATE LITE_WITH_X86 LITE_ON_TINY_PUBLISH)
    target_link_libraries(${TARGET}
        ${PADDLE_BINARY_DIR}/lite/api/libpaddle_light_api_shared.so
        m log atomic dl z)
    # gflags headers only (definitions resolve from the prebuilt shared lib).
    add_dependencies(${TARGET} extern_gflags)
    target_include_directories(${TARGET} PRIVATE ${THIRD_PARTY_PATH}/install/gflags/include)
    target_compile_options(${TARGET} BEFORE PRIVATE -Wno-ignored-qualifiers)
    if(NOT "${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
        add_custom_command(TARGET ${TARGET} POST_BUILD
            COMMAND "${CMAKE_STRIP}" -s "${TARGET}"
            COMMENT "Strip benchmark_bin debug symbols")
    endif()
else()
    lite_cc_binary(${TARGET} SRCS ${BENCHMARK_SRC}
                   DEPS gflags
                   CV_DEPS paddle_cv_arm)
endif()

if(ARM_TARGET_OS STREQUAL "android")
    if(NOT ("${ARM_TARGET_ARCH_ABI}" STREQUAL "x86_64" OR "${ARM_TARGET_ARCH_ABI}" STREQUAL "x86"))
    add_dependencies(${TARGET} extern_lite_download_${ILSVRC2012_1K_CLS_IMGS}_tar_gz
                               extern_lite_download_${ILSVRC2012_1K_CLS_LABELS}_tar_gz)
    string(REGEX REPLACE "[-%.]" "_" OPENCV_VERSION_EX ${OPENCV_VERSION})
    add_dependencies(${TARGET} extern_lite_download_${OPENCV_VERSION_EX}_tar_gz)
    include_directories(${OPENCV_INCLUDE_DIRS})
    target_link_libraries(${TARGET} ${OPENCV_LIBS} -lz)
    endif()
endif()