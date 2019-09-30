if (NOT LITE_ON_TINY_PUBLISH)
    add_subdirectory(pb)
endif()
add_subdirectory(cpp)
add_subdirectory(naive_buffer)

#lite_cc_library(runtime_lite SRCS runtime.cc)

#TODO(Superjomn) enable it again.
if(NOT LITE_ON_MOBILE AND NOT LITE_ON_TINY_PUBLISH)
    lite_cc_test(test_model_parser SRCS model_parser_test.cc
      DEPS model_parser framework_proto
      ARGS --model_dir=${LITE_MODEL_DIR}/lite_naive_model)
    if(WITH_TESTING)
        add_dependencies(test_model_parser extern_lite_download_lite_naive_model_tar_gz)
    endif(WITH_TESTING)
endif()

if (NOT LITE_ON_TINY_PUBLISH)
    lite_cc_library(compatible_pb SRCS compatible_pb.cc
      DEPS ${cpp_wrapper} ${naive_wrapper} ${pb_wrapper} framework_proto)
else()
    lite_cc_library(compatible_pb SRCS compatible_pb.cc DEPS ${cpp_wrapper} ${naive_wrapper})
endif()

lite_cc_library(model_parser SRCS model_parser.cc DEPS
    variable scope tensor scope
    target_wrapper_host
    compatible_pb
    memory
    CUDA_DEPS target_wrapper_cuda
    NPU_DEPS npu_helper)

lite_cc_test(test_compatible_pb SRCS compatible_pb_test.cc DEPS compatible_pb)
