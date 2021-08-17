if (NOT LITE_ON_TINY_PUBLISH)
  lite_cc_library (optimizer SRCS optimizer.cc DEPS mir_pass_manager model_parser program)
  add_subdirectory (mir)
endif()
