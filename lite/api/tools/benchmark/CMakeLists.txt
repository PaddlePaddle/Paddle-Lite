if(ARM_TARGET_OS STREQUAL "android")
    if (${ARM_TARGET_ARCH_ABI} STREQUAL "armv8")
        set(ARM_ABI "arm64-v8a")
    elseif(${ARM_TARGET_ARCH_ABI} STREQUAL "armv7")
        set(ARM_ABI "armeabi-v7a")
    endif()

    # OpenCV
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

    # Validation Dataset
    set(VALIDATION_DATASET_DIR "${THIRD_PARTY_PATH}/validation_dataset")
    set(VALIDATION_DATASET_URL "http://paddle-inference-dist.bj.bcebos.com/AI-Rank/validation_dataset")

    # ILSVRC 2012. We have extracted a minival subset which only contains 1000 clases.
    set(ILSVRC2012_1K_CLS_IMGS "ILSVRC2012_1000_cls")
    set(ILSVRC2012_1K_CLS_LABELS "ILSVRC2012_1000_cls_label_list")
    lite_download_and_uncompress(${VALIDATION_DATASET_DIR} ${VALIDATION_DATASET_URL} "${ILSVRC2012_1K_CLS_IMGS}.tar.gz")
    lite_download_and_uncompress(${VALIDATION_DATASET_DIR} ${VALIDATION_DATASET_URL} "${ILSVRC2012_1K_CLS_LABELS}.tar.gz")
endif()

FILE(GLOB_RECURSE BENCHMARK_SRC *.cc ../opt_base.cc)
if(NOT(ARM_TARGET_OS STREQUAL "android"))
    FILE(GLOB_RECURSE BENCHMARK_PRECISION_SRC precision_evaluation/*.cc)
    LIST(REMOVE_ITEM BENCHMARK_SRC ${BENCHMARK_PRECISION_SRC})
endif()
if(APPLE)
    message(STATUS "benchmark monitor is only available on android/linux os.")
    FILE(GLOB_RECURSE BENCHMARK_MONITOR_SRC profile/*.cc)
    LIST(REMOVE_ITEM BENCHMARK_SRC ${BENCHMARK_MONITOR_SRC})
endif()

set(TARGET "benchmark_bin")
lite_cc_binary(${TARGET} SRCS ${BENCHMARK_SRC}
               DEPS gflags
               CV_DEPS paddle_cv_arm)

if(ARM_TARGET_OS STREQUAL "android")
    # Validation Dataset
    add_dependencies(${TARGET} extern_lite_download_${ILSVRC2012_1K_CLS_IMGS}_tar_gz
                               extern_lite_download_${ILSVRC2012_1K_CLS_LABELS}_tar_gz)
    # OpenCV
    string(REGEX REPLACE "[-%.]" "_" OPENCV_VERSION_EX ${OPENCV_VERSION})
    add_dependencies(${TARGET} extern_lite_download_${OPENCV_VERSION_EX}_tar_gz)
    include_directories(${OPENCV_INCLUDE_DIRS})
    target_link_libraries(${TARGET} ${OPENCV_LIBS} -lz)
endif()
