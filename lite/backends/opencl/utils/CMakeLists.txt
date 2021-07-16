lite_cc_library(cl_cache SRCS cache.cc)
add_dependencies(cl_cache fbs_headers)
lite_cc_test(test_cl_cache SRCS test_cache.cc DEPS cl_cache)
