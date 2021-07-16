if (NOT LITE_WITH_METAL)
    return()
endif ()


FILE(GLOB LITE_METAL_KERNELS_SRC ${CMAKE_SOURCE_DIR}/lite/backends/metal/metal_kernel/*.metal
        ${CMAKE_SOURCE_DIR}/lite/backends/metal/metal_kernel/buffer/*.metal
        ${CMAKE_SOURCE_DIR}/lite/backends/metal/metal_kernel/texture/*.metal
        )

IF (DEFINED SDK_VERSION)
    #Defined by iOS toolchain
    SET(SDK_NAME "iphoneos")
    SET(TARGET_OPTION "-mios-version-min=${DEPLOYMENT_TARGET}")
ELSE ()
    SET(SDK_NAME "macosx")
    SET(TARGET_OPTION "")
ENDIF ()

add_custom_target(LiteMetalLIB
        DEPENDS ${CMAKE_BINARY_DIR}/lite.metallib
        COMMENT "Generating lite.metallib")

add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/lite.metallib
        COMMAND xcrun -sdk ${SDK_NAME} metal ${TARGET_OPTION} ${LITE_METAL_KERNELS_SRC} -o ${CMAKE_BINARY_DIR}/lite.metallib
        DEPENDS ${LITE_METAL_KERNELS_SRC}
        COMMENT "Built target lite.metallib")

find_library(METAL_LIBRARY Metal REQUIRED)
find_library(FOUNDATION_LIBRARY Foundation)

lite_cc_library(metal_target_wrapper SRCS
        context.mm
        metal_buffer.mm
        metal_context.mm
        metal_context_imp.mm
        metal_converter.mm
        metal_debug.mm
        metal_half.mm
        metal_image.mm
        mps_conv_datasource.m
        target_wrapper.mm
        )

target_link_libraries(metal_target_wrapper ${METAL_LIBRARY} ${FOUNDATION_LIBRARY})

add_dependencies(metal_target_wrapper LiteMetalLIB)
