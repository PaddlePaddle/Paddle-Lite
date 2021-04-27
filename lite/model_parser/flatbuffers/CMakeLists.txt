# flatbuffer data structure
lite_cc_library(fbs_wrapper SRCS
     op_desc.cc
     var_desc.cc
     block_desc.cc
     op_version_map.cc
     program_desc.cc
     param_desc.cc
     io.cc)
add_dependencies(fbs_wrapper fbs_headers)

lite_cc_test(test_vector_view SRCS vector_view_test.cc DEPS ${UNIT_TEST_DEPS})
lite_cc_test(test_fbs_io SRCS io_test.cc DEPS ${UNIT_TEST_DEPS})
lite_cc_test(test_program_desc SRCS program_desc_test.cc DEPS ${UNIT_TEST_DEPS})
