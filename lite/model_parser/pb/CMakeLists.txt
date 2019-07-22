lite_cc_library(var_desc SRCS var_desc.cc DEPS framework_proto)
lite_cc_library(op_desc SRCS op_desc.cc DEPS framework_proto)
lite_cc_library(block_desc SRCS block_desc.cc DEPS framework_proto)
lite_cc_library(program_desc SRCS program_desc.cc DEPS framework_proto)

set(pb_wrapper var_desc op_desc block_desc program_desc PARENT_SCOPE) 
