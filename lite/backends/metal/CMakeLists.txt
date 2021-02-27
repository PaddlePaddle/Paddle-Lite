if(NOT LITE_WITH_METAL)
    return()
endif()

FILE(GLOB LITE_METAL_KERNELS_SRC ${CMAKE_CURRENT_LIST_DIR}/metal_kernel/*.metal
    ${CMAKE_CURRENT_LIST_DIR}/metal_kernel/buffer/*.metal
    ${CMAKE_CURRENT_LIST_DIR}/metal_kernel/texture/*.metal
)
find_library(METAL_LIBRARY Metal REQUIRED)
find_library(FOUNDATION_LIBRARY Foundation)

#add_library(METAL_BASE FRAMEWORK IMPORTED)
#set_target_properties(METAL_BASE PROPERTIES IMPORTED_LOCATION ${METAL_LIBRARY})

#if(NOT Metal_FOUND)
#    message(FATAL_ERROR "Could not find Metal! ${METAL_LIBRARY}...${METAL_BASE}")
#endif()

IF(DEFINED SDK_VERSION)
    #Defined by iOS toolchain
    SET(SDK_NAME "iphoneos")
    SET(TARGET_OPTION "-target air64-apple-ios${DEPLOYMENT_TARGET}")
ELSE()
    SET(SDK_NAME "macosx")
    SET(TARGET_OPTION "")
ENDIF()


add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/lite.metallib COMMAND xcrun -sdk ${SDK_NAME} metal ${LITE_METAL_KERNELS_SRC} ${TARGET_OPTION} -o ${CMAKE_CURRENT_BINARY_DIR}/lite.metallib
 COMMENT "Built target lite.metallib")
add_custom_target (LiteMetalLIB DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/lite.metallib COMMENT "Generating lite.metallib")

add_library(
        metal_target_wrapper
        context.mm
        metal_context.mm
        metal_kernel.mm
        metal_device.mm
        metal_queue.mm
        metal_buffer.mm
        metal_image.mm
        metal_half.mm
        metal_converter.mm
        target_wrapper.mm
        metal_debug.mm)

target_link_libraries(metal_target_wrapper ${METAL_LIBRARY} ${FOUNDATION_LIBRARY})

add_dependencies(metal_target_wrapper LiteMetalLIB)
