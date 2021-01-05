lite_cc_library(cl_cache SRCS cache.cc DEPS model_base_io)
add_dependencies(cl_cache cl_cache_fbs_header)
lite_cc_test(test_cl_cache SRCS test_cache.cc DEPS cl_cache)
