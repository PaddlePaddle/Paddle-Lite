
set(dir ${CMAKE_CURRENT_SOURCE_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${dir}/build")

# gen test
ADD_EXECUTABLE(test-conv-op  operators/test_cov_op.cpp test_helper.h  test_include.h executor_for_test.h)
target_link_libraries(test-conv-op paddle-mobile)

# gen test
ADD_EXECUTABLE(test-mul-op  operators/test_mul_op.cpp test_helper.h  test_include.h)
target_link_libraries(test-mul-op paddle-mobile)

# gen test
ADD_EXECUTABLE(test-elementwiseadd-op  operators/test_elementwise_add_op.cpp test_helper.h  test_include.h)
target_link_libraries(test-elementwiseadd-op paddle-mobile)

# gen test
ADD_EXECUTABLE(test-concat-op  operators/test_concat_op.cpp test_helper.h  test_include.h)
target_link_libraries(test-concat-op paddle-mobile)

# gen test
ADD_EXECUTABLE(test-lrn-op  operators/test_lrn_op.cpp test_helper.h  test_include.h)
target_link_libraries(test-lrn-op paddle-mobile)

# gen test
ADD_EXECUTABLE(test-batchnorm-op  operators/test_batchnorm_op.cpp test_helper.h  test_include.h)
target_link_libraries(test-batchnorm-op paddle-mobile)

# gen test
ADD_EXECUTABLE(test-priorbox-op  operators/test_prior_box_op.cpp test_helper.h  test_include.h)
target_link_libraries(test-priorbox-op paddle-mobile)

# gen test
ADD_EXECUTABLE(test-boxcoder-op  operators/test_box_coder_op.cpp test_helper.h  test_include.h)
target_link_libraries(test-boxcoder-op paddle-mobile)

# gen test log
ADD_EXECUTABLE(test-log common/test_log.cpp)
target_link_libraries(test-log paddle-mobile)

# gen test log
ADD_EXECUTABLE(test-load framework/test_load.cpp)
target_link_libraries(test-load paddle-mobile)

# gen test log
# gen test
ADD_EXECUTABLE(test-optimize framework/test_optimize.cpp)
target_link_libraries(test-optimize paddle-mobile)


#gen test
ADD_EXECUTABLE(test-pool operators/test_pool_op.cpp test_helper.h test_include.h executor_for_test.h)
target_link_libraries(test-pool paddle-mobile)

#gen test
ADD_EXECUTABLE(test-softmax operators/test_softmax_op.cpp test_helper.h test_include.h executor_for_test.h)
target_link_libraries(test-softmax paddle-mobile)

# gen test
ADD_EXECUTABLE(test-gemm common/test_gemm.cpp)
target_link_libraries(test-gemm paddle-mobile)

# gen test
ADD_EXECUTABLE(test-enforce common/test_enforce.cpp)
target_link_libraries(test-enforce paddle-mobile)
