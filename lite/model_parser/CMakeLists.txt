# source code for unit tests
FILE(GLOB_RECURSE UNIT_TEST_SRC ${CMAKE_CURRENT_SOURCE_DIR}/*test*)

# protobuf parser source code
FILE(GLOB_RECURSE PB_PARSER_SRC ${CMAKE_CURRENT_SOURCE_DIR}/pb/*.cc)
LIST(REMOVE_ITEM PB_PARSER_SRC ${UNIT_TEST_SRC})

# flatbuffer parser source code
FILE(GLOB_RECURSE FLAT_PARSER_SRC ${CMAKE_CURRENT_SOURCE_DIR}/flatbuffers/*.cc)
LIST(REMOVE_ITEM FLAT_PARSER_SRC ${UNIT_TEST_SRC})

# ssa graph parser source code (this is used for removing circle struct from optimized model)
FILE(GLOB_RECURSE SSA_PARSER_SRC ${CMAKE_CURRENT_SOURCE_DIR}/ssa/*.cc)
LIST(REMOVE_ITEM SSA_PARSER_SRC ${UNIT_TEST_SRC})

# ssa graph parser source code (this is used for removing circle struct from optimized model)
FILE(GLOB_RECURSE NAIVE_PARSER_SRC ${CMAKE_CURRENT_SOURCE_DIR}/naive_buffer/*.cc)
LIST(REMOVE_ITEM NAIVE_PARSER_SRC ${UNIT_TEST_SRC})

# common code for core module
FILE(GLOB MODEL_PARSER_BASE_SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.cc)
LIST(REMOVE_ITEM MODEL_PARSER_BASE_SRC ${UNIT_TEST_SRC} ${CMAKE_CURRENT_SOURCE_DIR}/compatibility.cc)


set(MODEL_PARSER_SRC ${MODEL_PARSER_BASE_SRC} ${FLAT_PARSER_SRC})
set(MODEL_PAESER_DEPS "")


if (NOT LITE_ON_TINY_PUBLISH)
   set(MODEL_PARSER_SRC ${MODEL_PARSER_SRC} ${PB_PARSER_SRC} ${SSA_PARSER_SRC} ${NAIVE_PARSER_SRC})
   set(MODEL_PAESER_DEPS ${MODEL_PAESER_DEPS} framework_proto)
endif()

if (LITE_WITH_CUDA AND NOT LITE_ON_TINY_PUBLISH)
    set(MODEL_PARSER_SRC ${MODEL_PARSER_SRC} ${CMAKE_CURRENT_SOURCE_DIR}/compatibility.cc)
endif()


lite_cc_library(model_parser SRCS ${MODEL_PARSER_SRC} DEPS ${MODEL_PAESER_DEPS})
add_dependencies(model_parser fbs_headers)


add_subdirectory(naive_buffer)
add_subdirectory(flatbuffers)
add_subdirectory(ssa)

lite_cc_test(test_compatible_pb SRCS compatible_pb_test.cc)


#TODO(Superjomn) enable it again.
if(NOT LITE_ON_MOBILE AND NOT LITE_ON_TINY_PUBLISH)
    lite_cc_test(test_model_parser SRCS model_parser_test.cc
      ARGS --model_dir=${LITE_MODEL_DIR}/lite_naive_model)
    if(WITH_TESTING)
        add_dependencies(test_model_parser extern_lite_download_lite_naive_model_tar_gz)
    endif(WITH_TESTING)
endif()


if (LITE_WITH_CUDA AND NOT LITE_ON_TINY_PUBLISH)
    lite_cc_test(test_compatibility SRCS compatibility_test.cc)
endif()
