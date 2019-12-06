if (NOT LITE_WITH_BM)
    return()
endif()

lite_cc_library(target_wrapper_bm SRCS target_wrapper.cc bm_context.cc DEPS ${bm_runtime_libs})
