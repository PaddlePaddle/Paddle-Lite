set(dir ${CMAKE_CURRENT_SOURCE_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${dir}/build")

if (googlenet)
    # gen test
    ADD_EXECUTABLE(test-googlenet net/test_googlenet.cpp test_helper.h  test_include.h executor_for_test.h)
    target_link_libraries(test-googlenet paddle-mobile)
elseif (mobilenet)
    # gen test
    ADD_EXECUTABLE(test-mobilenet net/test_mobilenet.cpp test_helper.h  test_include.h executor_for_test.h)
    target_link_libraries(test-mobilenet paddle-mobile)
elseif (yolo)
    # gen test
    ADD_EXECUTABLE(test-yolo net/test_yolo.cpp test_helper.h  test_include.h executor_for_test.h)
    target_link_libraries(test-yolo paddle-mobile)
elseif (squeezenet)
    # gen test
    ADD_EXECUTABLE(test-squeezenet net/test_squeezenet.cpp test_helper.h  test_include.h executor_for_test.h)
    target_link_libraries(test-squeezenet paddle-mobile)
elseif(resnet)
    # gen test
    ADD_EXECUTABLE(test-resnet net/test_resnet.cpp test_helper.h  test_include.h executor_for_test.h)
    target_link_libraries(test-resnet paddle-mobile)
else ()

    # gen test
    ADD_EXECUTABLE(test-resnet net/test_resnet.cpp test_helper.h  test_include.h executor_for_test.h)
    target_link_libraries(test-resnet paddle-mobile)

    # gen test
    ADD_EXECUTABLE(test-squeezenet net/test_squeezenet.cpp test_helper.h  test_include.h executor_for_test.h)
    target_link_libraries(test-squeezenet paddle-mobile)

    # gen test
    ADD_EXECUTABLE(test-yolo net/test_yolo.cpp test_helper.h  test_include.h executor_for_test.h)
    target_link_libraries(test-yolo paddle-mobile)

    # gen test
    ADD_EXECUTABLE(test-googlenet net/test_googlenet.cpp test_helper.h  test_include.h executor_for_test.h)
    target_link_libraries(test-googlenet paddle-mobile)

    # gen test
    ADD_EXECUTABLE(test-conv-op  operators/test_cov_op.cpp test_helper.h  test_include.h executor_for_test.h)
    target_link_libraries(test-conv-op paddle-mobile)

    # gen test
    ADD_EXECUTABLE(test-mul-op  operators/test_mul_op.cpp test_helper.h  test_include.h)
    target_link_libraries(test-mul-op paddle-mobile)

    # gen test
    ADD_EXECUTABLE(test-elementwiseadd-op operators/test_elementwise_add_op.cpp test_helper.h  test_include.h)
    target_link_libraries(test-elementwiseadd-op paddle-mobile)

    # gen test
    ADD_EXECUTABLE(test-concat-op operators/test_concat_op.cpp test_helper.h  test_include.h)
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

    # gen test
    ADD_EXECUTABLE(test-transpose-op  operators/test_transpose_op.cpp test_helper.h  test_include.h)
    target_link_libraries(test-transpose-op paddle-mobile)

    # gen test
    ADD_EXECUTABLE(test-multiclassnms-op  operators/test_multiclass_nms_op.cpp test_helper.h  test_include.h)
    target_link_libraries(test-multiclassnms-op paddle-mobile)

    # gen test
    ADD_EXECUTABLE(test-reshape-op  operators/test_reshape_op.cpp test_helper.h  test_include.h)
    target_link_libraries(test-reshape-op paddle-mobile)

    # gen test
    ADD_EXECUTABLE(test-relu-op  operators/test_relu_op.cpp test_helper.h  test_include.h)
    target_link_libraries(test-relu-op paddle-mobile)

    # gen test
    ADD_EXECUTABLE(test-fc-op  operators/test_fushion_fc_op.cpp test_helper.h  test_include.h)
    target_link_libraries(test-fc-op paddle-mobile)

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

    # gen test - test if openmp works
    ADD_EXECUTABLE(test-openmp common/test_openmp.cpp test_helper.h test_include.h executor_for_test.h)
    target_link_libraries(test-openmp paddle-mobile)

    # gen test
    ADD_EXECUTABLE(test-mobilenetssd net/test_mobilenet+ssd.cpp test_helper.h  test_include.h executor_for_test.h)
    target_link_libraries(test-mobilenetssd paddle-mobile)

    # gen test
    ADD_EXECUTABLE(test-sigmoid operators/test_sigmoid_op.cpp  test_include.h)
    target_link_libraries(test-sigmoid paddle-mobile)

    # gen test
    ADD_EXECUTABLE(test-depthwise-conv-op operators/test_depthwise_conv_op.cpp test_helper.h test_include.h executor_for_test.h)
    target_link_libraries(test-depthwise-conv-op paddle-mobile)

    # gen test
    ADD_EXECUTABLE(test-mobilenet net/test_mobilenet.cpp test_helper.h  test_include.h executor_for_test.h)
    target_link_libraries(test-mobilenet paddle-mobile)

    #add_library(test-lib-size SHARED common/test_lib_size.h common/test_lib_size.cpp)

endif()
