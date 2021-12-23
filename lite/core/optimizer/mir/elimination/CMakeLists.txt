if (NOT LITE_WITH_ARM)
  # NOTE disabled for the proto_desc is not valid yet.
  # TODO(Superjomn) enable them if valid latter.
  # lite_cc_test(test_identity_scale_eliminate_pass
  #   SRCS identity_scale_eliminate_pass_test.cc
  #   DEPS core proto_desc cpp_op_desc
  #   ops
  #   )
endif()
 
