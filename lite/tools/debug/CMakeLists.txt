if(NOT LITE_ON_MODEL_OPTIMIZE_TOOL)
    return()
endif()

lite_cc_library(debug_utils SRCS debug_utils.cc DEPS ops model_parser)
lite_cc_binary(lite_model_debug_tool SRCS model_debug_tool.cc
    DEPS debug_utils gflags)
