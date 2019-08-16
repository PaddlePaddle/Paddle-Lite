lite_cc_library(cpp_op_desc SRCS op_desc.cc DEPS any)
lite_cc_library(cpp_var_desc SRCS var_desc.cc)
lite_cc_library(cpp_block_desc SRCS block_desc.cc)
lite_cc_library(cpp_program_desc SRCS program_desc.cc)

set(cpp_wrapper cpp_op_desc cpp_var_desc cpp_block_desc cpp_program_desc PARENT_SCOPE)
