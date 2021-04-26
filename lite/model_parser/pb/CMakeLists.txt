# protobuf data struct
lite_cc_library(pb_wrapper SRCS
  program_desc.cc
  block_desc.cc
  op_desc.cc
  op_version_map.cc
  tensor_io.cc
  param_desc.cc
  var_desc.cc
  utils.cc
  DEPS framework_proto
)
