add_subdirectory(fusion)
add_subdirectory(elimination)
add_subdirectory(subgraph)
lite_cc_test(test_pattern_matcher SRCS pattern_matcher_test.cc DEPS core)
# for mobile, unnecessary to compile the following testings.
if(LITE_WITH_ARM)
    return()
endif()
lite_cc_test(test_mir_pass_manager SRCS pass_manager_test.cc DEPS core)
