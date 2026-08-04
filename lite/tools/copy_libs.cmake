
# Helper script to robustly copy library artifacts for Paddle Lite publish stage
function(copy_if_exists SRC_PATH DEST_DIR)
    if(EXISTS "${SRC_PATH}")
        message(STATUS "Copying ${SRC_PATH} to ${DEST_DIR}")
        file(COPY "${SRC_PATH}" DESTINATION "${DEST_DIR}")
    endif()
endfunction()

# 1. Bundled static libraries (Try BOTH root and lite/api)
copy_if_exists("${BINARY_DIR}/libpaddle_api_full_bundled.a" "${DEST_DIR}")
copy_if_exists("${BINARY_DIR}/lite/api/libpaddle_api_full_bundled.a" "${DEST_DIR}")
copy_if_exists("${BINARY_DIR}/libpaddle_api_light_bundled.a" "${DEST_DIR}")
copy_if_exists("${BINARY_DIR}/lite/api/libpaddle_api_light_bundled.a" "${DEST_DIR}")

# 2. Shared libraries (GLOB both locations)
file(GLOB SO_LIBS_ROOT "${BINARY_DIR}/*.so")
file(GLOB SO_LIBS_API "${BINARY_DIR}/lite/api/*.so")
if(SO_LIBS_ROOT)
    file(COPY ${SO_LIBS_ROOT} DESTINATION "${DEST_DIR}")
endif()
if(SO_LIBS_API)
    file(COPY ${SO_LIBS_API} DESTINATION "${DEST_DIR}")
endif()

# 3. macOS/iOS Dylibs
file(GLOB DYLIBS_ROOT "${BINARY_DIR}/*.dylib")
file(GLOB DYLIBS_API "${BINARY_DIR}/lite/api/*.dylib")
if(DYLIBS_ROOT)
    file(COPY ${DYLIBS_ROOT} DESTINATION "${DEST_DIR}")
endif()
if(DYLIBS_API)
    file(COPY ${DYLIBS_API} DESTINATION "${DEST_DIR}")
endif()
