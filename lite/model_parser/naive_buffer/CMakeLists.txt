if (LITE_ON_TINY_PUBLISH)
  set(naive_wrapper "")
  return()
endif()

lite_cc_library(naive_buffer SRCS naive_buffer.cc DEPS types)

add_subdirectory(proto)

lite_cc_library(nb_op_desc SRCS op_desc.cc DEPS framework_nb)
lite_cc_library(nb_var_desc SRCS var_desc.cc DEPS framework_nb)
lite_cc_library(nb_op_version_map SRCS op_version_map.cc DEPS framework_nb)
lite_cc_library(nb_param_desc SRCS param_desc.cc DEPS framework_nb)
lite_cc_library(nb_combined_params_desc SRCS combined_params_desc.cc DEPS nb_param_desc framework_nb)
lite_cc_library(nb_block_desc SRCS block_desc.cc DEPS framework_nb)
lite_cc_library(nb_program_desc SRCS program_desc.cc DEPS framework_nb)

set(naive_wrapper 
    nb_op_desc nb_var_desc nb_param_desc nb_combined_params_desc
    nb_block_desc nb_program_desc nb_op_version_map PARENT_SCOPE)

if(NOT WITH_COVERAGE)
    lite_cc_test(test_naive_buffer SRCS naive_buffer_test.cc DEPS naive_buffer)
endif()
lite_cc_test(test_naive_buffer_wrapper SRCS naive_buffer_wrapper_test.cc 
             DEPS nb_op_desc nb_var_desc nb_param_desc nb_combined_params_desc 
             nb_block_desc nb_program_desc)
