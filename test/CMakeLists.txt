set(dir ${CMAKE_CURRENT_SOURCE_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${dir}/build")
set(FOUND_MATCH OFF)

set(CON -1)

message(STATUS "nets :${NET}")

list(FIND NET "googlenet" CON)
if (CON GREATER -1)
    # gen test
    ADD_EXECUTABLE(test-googlenet net/test_googlenet.cpp test_helper.h test_include.h executor_for_test.h)
    target_link_libraries(test-googlenet paddle-mobile)

    # gen test
    ADD_EXECUTABLE(test-googlenet-quali net/test_googlenet_quali.cpp test_helper.h test_include.h executor_for_test.h)
    target_link_libraries(test-googlenet-quali paddle-mobile)
    set(FOUND_MATCH ON)

endif ()

list(FIND NET "mobilenet" CON)
if (CON GREATER -1)
    # gen test
    ADD_EXECUTABLE(test-mobilenet net/test_mobilenet.cpp test_helper.h test_include.h executor_for_test.h)
    target_link_libraries(test-mobilenet paddle-mobile)

    # gen test
    ADD_EXECUTABLE(test-mobilenet-combine net/test_mobilenet_combine.cpp test_helper.h test_include.h executor_for_test.h)
    target_link_libraries(test-mobilenet-combine paddle-mobile)
    set(FOUND_MATCH ON)

endif ()

list(FIND NET "yolo" CON)
if (CON GREATER -1)
    # gen test
    ADD_EXECUTABLE(test-yolo net/test_yolo.cpp test_helper.h test_include.h executor_for_test.h)
    target_link_libraries(test-yolo paddle-mobile)
    # gen test
    ADD_EXECUTABLE(test-yolo-combined net/test_yolo_combined.cpp test_helper.h test_include.h executor_for_test.h)
    target_link_libraries(test-yolo-combined paddle-mobile)
    set(FOUND_MATCH ON)

endif ()

list(FIND NET "squeezenet" CON)
if (CON GREATER -1)
    # gen test
    ADD_EXECUTABLE(test-squeezenet net/test_squeezenet.cpp test_helper.h test_include.h executor_for_test.h)
    target_link_libraries(test-squeezenet paddle-mobile)
    set(FOUND_MATCH ON)

endif ()

list(FIND NET "resnet" CON)
if (CON GREATER -1)
    # gen test
    ADD_EXECUTABLE(test-resnet net/test_resnet.cpp test_helper.h test_include.h executor_for_test.h)
    target_link_libraries(test-resnet paddle-mobile)
    set(FOUND_MATCH ON)

endif ()

list(FIND NET "FPGA_NET_V1" CON)
if (CON GREATER -1)
    ADD_EXECUTABLE(test-resnet50 fpga/test_resnet50.cpp test_helper.h test_include.h executor_for_test.h)
    target_link_libraries(test-resnet50 paddle-mobile)

    ADD_EXECUTABLE(test-densebox net/test_densebox_combine.cpp test_helper.h test_include.h executor_for_test.h)
    target_link_libraries(test-densebox paddle-mobile)

    set(FOUND_MATCH ON)
endif ()

list(FIND NET "FPGA_NET_V2" CON)
if (CON GREATER -1)
    ADD_EXECUTABLE(test-resnet50 fpga/test_resnet50.cpp test_helper.h test_include.h executor_for_test.h)
    target_link_libraries(test-resnet50 paddle-mobile)

    ADD_EXECUTABLE(test-pe fpga/test_pe.cpp)
    target_link_libraries(test-pe paddle-mobile)

    ADD_EXECUTABLE(test-densebox net/test_densebox_combine.cpp test_helper.h test_include.h executor_for_test.h)
    target_link_libraries(test-densebox paddle-mobile)

    set(FOUND_MATCH ON)
endif ()

list(FIND NET "mobilenetssd" CON)
if (CON GREATER -1)
    # gen test
    ADD_EXECUTABLE(test-mobilenetssd net/test_mobilenet+ssd.cpp test_helper.h test_include.h executor_for_test.h)
    target_link_libraries(test-mobilenetssd paddle-mobile)

    set(FOUND_MATCH ON)

endif ()

list(FIND NET "nlp" CON)
if (CON GREATER -1)
    # gen test
    ADD_EXECUTABLE(test-nlp net/test_nlp.cpp test_helper.h test_include.h executor_for_test.h)
    target_link_libraries(test-nlp paddle-mobile)

    # gen test
    ADD_EXECUTABLE(test-gru-op operators/test_gru_op.cpp test_helper.h test_include.h)
    target_link_libraries(test-gru-op paddle-mobile)
    set(FOUND_MATCH ON)

endif ()

list(FIND NET "mobilenetfssd" CON)
if (CON GREATER -1)
    # gen test
    ADD_EXECUTABLE(test-fssd net/test_mobilenet_025_fssd.cpp test_helper.h test_include.h)
    target_link_libraries(test-fssd paddle-mobile)

    set(FOUND_MATCH ON)

endif ()

list(FIND NET "genet" CON)
if (CON GREATER -1)
    # gen test
    ADD_EXECUTABLE(test-genet net/test_genet_combine.cpp test_helper.h test_include.h executor_for_test.h)
    target_link_libraries(test-genet paddle-mobile)
    set(FOUND_MATCH ON)

endif ()

if (NOT FOUND_MATCH)
    # gen test
    ADD_EXECUTABLE(test-resnet net/test_resnet.cpp test_helper.h test_include.h executor_for_test.h)
    target_link_libraries(test-resnet paddle-mobile)

    # gen test
    ADD_EXECUTABLE(test-squeezenet net/test_squeezenet.cpp test_helper.h test_include.h executor_for_test.h)
    target_link_libraries(test-squeezenet paddle-mobile)

    # gen test
    ADD_EXECUTABLE(test-yolo net/test_yolo.cpp test_helper.h test_include.h executor_for_test.h)
    target_link_libraries(test-yolo paddle-mobile)

    # gen test
    ADD_EXECUTABLE(test_yolo_combined net/test_yolo_combined.cpp test_helper.h test_include.h executor_for_test.h)
    target_link_libraries(test_yolo_combined paddle-mobile)

    # gen test
    ADD_EXECUTABLE(test-googlenet net/test_googlenet.cpp test_helper.h test_include.h executor_for_test.h)
    target_link_libraries(test-googlenet paddle-mobile)

    # gen test
    ADD_EXECUTABLE(test-googlenet-quali net/test_googlenet_quali.cpp test_helper.h test_include.h executor_for_test.h)
    target_link_libraries(test-googlenet-quali paddle-mobile)

    # gen test
    ADD_EXECUTABLE(test-conv-op operators/test_cov_op.cpp test_helper.h test_include.h executor_for_test.h)
    target_link_libraries(test-conv-op paddle-mobile)

    # gen test
    ADD_EXECUTABLE(test-mul-op operators/test_mul_op.cpp test_helper.h test_include.h)
    target_link_libraries(test-mul-op paddle-mobile)

    # gen test
    ADD_EXECUTABLE(test-elementwiseadd-op operators/test_elementwise_add_op.cpp test_helper.h test_include.h)
    target_link_libraries(test-elementwiseadd-op paddle-mobile)

    # gen test
    ADD_EXECUTABLE(test-elementwisesub-op operators/test_elementwise_sub_op.cpp test_helper.h test_include.h)
    target_link_libraries(test-elementwisesub-op paddle-mobile)

    # gen test
    ADD_EXECUTABLE(test-im2sequence-op operators/test_im2sequence_op.cpp test_helper.h test_include.h)
    target_link_libraries(test-im2sequence-op paddle-mobile)

	# gen test
    ADD_EXECUTABLE(test-concat-op operators/test_concat_op.cpp test_helper.h test_include.h)
    target_link_libraries(test-concat-op paddle-mobile)

    # gen test
    ADD_EXECUTABLE(test-lrn-op operators/test_lrn_op.cpp test_helper.h test_include.h)
    target_link_libraries(test-lrn-op paddle-mobile)

    # gen test
    ADD_EXECUTABLE(test-batchnorm-op operators/test_batchnorm_op.cpp test_helper.h test_include.h)
    target_link_libraries(test-batchnorm-op paddle-mobile)

    # gen test
    ADD_EXECUTABLE(test-priorbox-op operators/test_prior_box_op.cpp test_helper.h test_include.h)
    target_link_libraries(test-priorbox-op paddle-mobile)

    # gen test
    ADD_EXECUTABLE(test-boxcoder-op operators/test_box_coder_op.cpp test_helper.h test_include.h)
    target_link_libraries(test-boxcoder-op paddle-mobile)

    # gen test
    ADD_EXECUTABLE(test-transpose-op operators/test_transpose_op.cpp test_helper.h test_include.h)
    target_link_libraries(test-transpose-op paddle-mobile)

    # gen test
    ADD_EXECUTABLE(test-transpose2-op operators/test_transpose2_op.cpp test_helper.h test_include.h)
    target_link_libraries(test-transpose2-op paddle-mobile)

    # gen test
    ADD_EXECUTABLE(test-multiclassnms-op operators/test_multiclass_nms_op.cpp test_helper.h test_include.h)
    target_link_libraries(test-multiclassnms-op paddle-mobile)

    # gen test
    ADD_EXECUTABLE(test-polygon-box-transform-op operators/test_polygon_box_transform_op.cpp test_helper.h test_include.h)
    target_link_libraries(test-polygon-box-transform-op paddle-mobile)

    # gen test
    ADD_EXECUTABLE(test-fill-constant-op operators/test_fill_constant_op.cpp test_helper.h test_include.h)
    target_link_libraries(test-fill-constant-op paddle-mobile)

    # gen test
    ADD_EXECUTABLE(test-reshape-op operators/test_reshape_op.cpp test_helper.h test_include.h)
    target_link_libraries(test-reshape-op paddle-mobile)

    # gen test
    ADD_EXECUTABLE(test-reshape2-op operators/test_reshape2_op.cpp test_helper.h test_include.h)
    target_link_libraries(test-reshape2-op paddle-mobile)

    # gen test
    ADD_EXECUTABLE(test-relu-op operators/test_relu_op.cpp test_helper.h test_include.h)
    target_link_libraries(test-relu-op paddle-mobile)

    # gen test
    ADD_EXECUTABLE(test-fc-op operators/test_fusion_fc_op.cpp test_helper.h test_include.h)
    target_link_libraries(test-fc-op paddle-mobile)

    # gen test
    ADD_EXECUTABLE(test-sum-op operators/test_sum_op.cpp test_helper.h test_include.h)
    target_link_libraries(test-sum-op paddle-mobile)

    # test quantize op
    ADD_EXECUTABLE(test-quantize-op operators/test_quantize_op.cpp test_helper.h test_include.h)
    target_link_libraries(test-quantize-op paddle-mobile)

    # test dequantize op
    ADD_EXECUTABLE(test-dequantize-op operators/test_dequantize_op.cpp test_helper.h test_include.h)
    target_link_libraries(test-dequantize-op paddle-mobile)

    # test int8 conv op
    ADD_EXECUTABLE(test-int8-conv-op operators/test_int8_conv_op.cpp test_helper.h test_include.h)
    target_link_libraries(test-int8-conv-op paddle-mobile)

    # gen test log
    ADD_EXECUTABLE(test-log common/test_log.cpp)
    target_link_libraries(test-log paddle-mobile)

    # gen test log
    ADD_EXECUTABLE(test-load framework/test_load.cpp)
    target_link_libraries(test-load paddle-mobile)

    # gen test log
    ADD_EXECUTABLE(test-loadmemory framework/test_load_memory.cpp)
    target_link_libraries(test-loadmemory paddle-mobile)

    # gen test log
    ADD_EXECUTABLE(test-loadmemory-inference framework/test_load_memory_inference_api.cpp)
    target_link_libraries(test-loadmemory-inference paddle-mobile)

    ADD_EXECUTABLE(test-inference-api framework/test_inference_api.cpp)
    target_link_libraries(test-inference-api paddle-mobile)


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
    ADD_EXECUTABLE(test-gemm-accuracy common/test_gemm_accuracy.cpp)
    target_link_libraries(test-gemm-accuracy paddle-mobile)

    # gen test
    ADD_EXECUTABLE(test-gemm-int8-accuracy common/test_gemm_int8_accuracy.cpp)
    target_link_libraries(test-gemm-int8-accuracy paddle-mobile)

    # gen test
    ADD_EXECUTABLE(test-gemm-perf common/test_gemm_perf.cpp)
    target_link_libraries(test-gemm-perf paddle-mobile)

    # gen test
    ADD_EXECUTABLE(test-enforce common/test_enforce.cpp)
    target_link_libraries(test-enforce paddle-mobile)

    # gen test - test if openmp works
    ADD_EXECUTABLE(test-openmp common/test_openmp.cpp test_helper.h test_include.h executor_for_test.h)
    target_link_libraries(test-openmp paddle-mobile)

    # gen test
    ADD_EXECUTABLE(test-mobilenetssd net/test_mobilenet+ssd.cpp test_helper.h test_include.h executor_for_test.h)
    target_link_libraries(test-mobilenetssd paddle-mobile)

    # gen test
    ADD_EXECUTABLE(test-mobilenet-combine net/test_mobilenet_combine.cpp test_helper.h test_include.h executor_for_test.h)
    target_link_libraries(test-mobilenet-combine paddle-mobile)

    # gen test
    ADD_EXECUTABLE(test-genet net/test_genet_combine.cpp test_helper.h test_include.h executor_for_test.h)
    target_link_libraries(test-genet paddle-mobile)

    # gen test
    ADD_EXECUTABLE(test-sigmoid operators/test_sigmoid_op.cpp test_include.h)
    target_link_libraries(test-sigmoid paddle-mobile)

    # gen test
    ADD_EXECUTABLE(test-depthwise-conv-op operators/test_depthwise_conv_op.cpp test_helper.h test_include.h executor_for_test.h)
    target_link_libraries(test-depthwise-conv-op paddle-mobile)

    # gen test
    ADD_EXECUTABLE(test-mobilenet net/test_mobilenet.cpp test_helper.h test_include.h executor_for_test.h)
    target_link_libraries(test-mobilenet paddle-mobile)

    # gen test
    ADD_EXECUTABLE(test-conv-add-relu-op operators/test_conv_add_relu_op.cpp test_helper.h test_include.h executor_for_test.h)
    target_link_libraries(test-conv-add-relu-op paddle-mobile)

    # gen test
    ADD_EXECUTABLE(test-conv-add-bn-relu-op operators/test_fusion_conv_add_bn_relu_op.cpp test_helper.h test_include.h executor_for_test.h)
    target_link_libraries(test-conv-add-bn-relu-op paddle-mobile)

    # gen test
    ADD_EXECUTABLE(test-nlp net/test_nlp.cpp test_helper.h test_include.h executor_for_test.h)
    target_link_libraries(test-nlp paddle-mobile)

    # gen test
    ADD_EXECUTABLE(test-gru-op operators/test_gru_op.cpp test_helper.h test_include.h)
    target_link_libraries(test-gru-op paddle-mobile)

    # gen test

    ADD_EXECUTABLE(test-inceptionv4 net/test_inceptionv4.cpp test_helper.h test_include.h executor_for_test.h)
    target_link_libraries(test-inceptionv4 paddle-mobile)

    # gen test
    ADD_EXECUTABLE(test-alexnet net/test_alexnet.cpp test_helper.h test_include.h executor_for_test.h)
    target_link_libraries(test-alexnet paddle-mobile)

    ADD_EXECUTABLE(test-googlenetv1 net/test_googlenetv1_combine.cpp test_helper.h test_include.h)
    target_link_libraries(test-googlenetv1 paddle-mobile)

    # gen test
    ADD_EXECUTABLE(test-fssd net/test_mobilenet_025_fssd.cpp test_helper.h test_include.h)
    target_link_libraries(test-fssd paddle-mobile)

    # gen test
    ADD_EXECUTABLE(test-mobilenetgpu  net/test_mobilenet_GPU.cpp test_helper.h  test_include.h)
    target_link_libraries(test-mobilenetgpu paddle-mobile)

    # gen test
    ADD_EXECUTABLE(test-yologpu net/test_yologpu.cpp test_helper.h  test_include.h executor_for_test.h)
    target_link_libraries(test-yologpu paddle-mobile)

    # gen test
    ADD_EXECUTABLE(test-multi-process net/test_multi_inference_predict.cpp test_helper.h test_include.h)
    target_link_libraries(test-multi-process paddle-mobile)

    # gen test
    ADD_EXECUTABLE(test-eng net/test_eng.cpp test_helper.h test_include.h)
    target_link_libraries(test-eng paddle-mobile)

    #add_library(test-lib-size SHARED common/test_lib_size.h common/test_lib_size.cpp)
endif ()
