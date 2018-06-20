IF(PADDLE_MOBILE_MALI_GPU)
   cc_library(acl_ops SRCS acl_operator.cc acl_tensor.cc)
ENDIF()

