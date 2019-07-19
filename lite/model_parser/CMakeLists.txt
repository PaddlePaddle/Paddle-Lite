add_subdirectory(pb)
add_subdirectory(cpp)
add_subdirectory(naive_buffer)

#lite_cc_library(runtime_lite SRCS runtime.cc)

#TODO(Superjomn) enable it again.
if(NOT LITE_ON_MOBILE)
    lite_cc_test(test_model_parser SRCS model_parser_test.cc
      DEPS model_parser framework_proto
      ARGS --model_dir=${LITE_MODEL_DIR}/lite_naive_model)
    if(WITH_TESTING)
        add_dependencies(test_model_parser extern_lite_download_lite_naive_model_tar_gz)
    endif(WITH_TESTING)
endif()

lite_cc_library(compatible_pb SRCS compatible_pb.cc
  DEPS op_desc framework_proto var_desc cpp_op_desc nb_op_desc)

lite_cc_library(model_parser SRCS model_parser.cc DEPS
    variable scope tensor scope
    target_wrapper_host
    compatible_pb
    memory
    ${naive_wrapper}
    CUDA_DEPS target_wrapper_cuda)

lite_cc_test(test_op_desc SRCS op_desc_test.cc DEPS cpp_op_desc op_desc nb_op_desc compatible_pb)
