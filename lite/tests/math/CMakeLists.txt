if((WITH_TESTING OR WITH_MATH_TESTING) AND LITE_WITH_ARM AND ANDROID)
    FILE(GLOB TEST_MATH_SRC_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*.cc")
    message(STATUS "math ut files: ${TEST_MATH_SRC_FILES}")
    add_compile_options(-fexceptions)
    # build test cases
    foreach(SRC_NAME ${TEST_MATH_SRC_FILES})
        #unpack the dir "/"
        string(REPLACE "/" ";" FILE_LIST ${SRC_NAME})
        list(GET FILE_LIST -1 TEST_CASE_NAME)
        #get the file name without suffix
        string(REPLACE "." ";" FILE_LIST ${TEST_CASE_NAME})
        list(GET FILE_LIST 0 TEST_CASE_NAME)
        add_executable(${TEST_CASE_NAME}  ${SRC_NAME})
        target_link_libraries(${TEST_CASE_NAME} math_arm)
        set_target_properties(${TEST_CASE_NAME} PROPERTIES
                RUNTIME_OUTPUT_DIRECTORY
                ${CMAKE_INSTALL_PREFIX}/unit_test)
        add_custom_command(TARGET ${TEST_CASE_NAME} POST_BUILD
                COMMAND "${CMAKE_STRIP}" -s
                "${CMAKE_INSTALL_PREFIX}/unit_test/${TEST_CASE_NAME}"
                COMMENT "Strip debug symbols done on final executable file.")
    endforeach()
    # lite_cc_test(test_sgemm SRCS test_sgemm.cc DEPS math_arm)
endif()
