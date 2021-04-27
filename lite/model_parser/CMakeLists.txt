if (NOT LITE_ON_TINY_PUBLISH)
    add_subdirectory(pb)
    add_subdirectory(naive_buffer)
endif()
add_subdirectory(base)
add_subdirectory(general)
add_subdirectory(flatbuffers)

# naive-buffer and protobuf are not applicable except that
# it's not in tiny_publish mode
if (NOT LITE_ON_TINY_PUBLISH)
    set(model_parser_deps pb_wrapper framework_proto naive_wrapper)
endif()

lite_cc_library(model_parser SRCS
                model_parser.cc # model parser defination
                compatible_pb.cc # conversation rules between fbs/naive/cpp/pb
                DEPS base_wrapper cpp_wrapper fbs_wrapper ${model_parser_deps})


#TODO(Superjomn) enable it again.
if(NOT LITE_ON_MOBILE AND NOT LITE_ON_TINY_PUBLISH)
    lite_cc_test(test_model_parser SRCS model_parser_test.cc
      DEPS ${UNIT_TEST_DEPS} framework_proto
      ARGS --model_dir=${LITE_MODEL_DIR}/lite_naive_model)
    if(WITH_TESTING)
        add_dependencies(test_model_parser extern_lite_download_lite_naive_model_tar_gz)
    endif(WITH_TESTING)
endif()
