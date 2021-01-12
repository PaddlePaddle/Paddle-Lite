function(lite_fbs_library TARGET)
  set(multiValueArgs SRCS FBS_DEPS)
  cmake_parse_arguments(args "" "" "${multiValueArgs}" ${ARGN})
  lite_cc_library(${TARGET} SRCS ${args_SRCS})
  add_dependencies(${TARGET} ${args_FBS_DEPS})
endfunction()
lite_fbs_library(fbs_wrapper 
            SRCS
                op_desc.cc
                var_desc.cc
                block_desc.cc
                op_version_map.cc
                program_desc.cc
                param_desc.cc
            FBS_DEPS
                fbs_headers
                model_base_io)

#lite_cc_library(fbs_io SRCS io.cc DEPS fbs_program_desc fbs_param_desc scope model_base_io)
lite_cc_test(test_vector_view SRCS vector_view_test.cc DEPS fbs_wrapper)
lite_cc_test(test_fbs_io SRCS io_test.cc DEPS fbs_io fbs_wrapper)
lite_cc_test(test_program_desc SRCS program_desc_test.cc DEPS fbs_wrapper)
