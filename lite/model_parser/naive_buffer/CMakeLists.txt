# naive buffer data struct
# help: naive buffer is the optimized model data struct before Paddle-Lite v2.6.0
lite_cc_library(naive_wrapper SRCS
                op_desc.cc
                var_desc.cc
                op_version_map.cc
                param_desc.cc
                combined_params_desc.cc
                block_desc.cc
                program_desc.cc
                naive_buffer.cc
                proto/framework.nb.cc)


if(NOT WITH_COVERAGE)
    lite_cc_test(test_naive_buffer SRCS naive_buffer_test.cc DEPS ${UNIT_TEST_DEPS})
endif()
lite_cc_test(test_naive_buffer_wrapper SRCS naive_buffer_wrapper_test.cc 
             DEPS ${UNIT_TEST_DEPS})
