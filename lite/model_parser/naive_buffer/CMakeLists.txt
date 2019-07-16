lite_cc_library(naive_buffer_lite SRCS naive_buffer.cc DEPS types_lite)

add_subdirectory(proto)

lite_cc_library(nb_op_desc_lite SRCS op_desc.cc DEPS framework_nb_lite)
lite_cc_library(nb_var_desc_lite SRCS var_desc.cc DEPS framework_nb_lite)
lite_cc_library(nb_param_desc_lite SRCS param_desc.cc DEPS framework_nb_lite)

set(naive_wrapper_lite nb_op_desc_lite nb_var_desc_lite nb_param_desc_lite PARENT_SCOPE)

lite_cc_test(test_naive_buffer_lite SRCS naive_buffer_test.cc DEPS naive_buffer_lite)
lite_cc_test(test_naive_buffer_wrapper_lite SRCS naive_buffer_wrapper_test.cc 
             DEPS nb_op_desc_lite nb_var_desc_lite nb_param_desc_lite)
