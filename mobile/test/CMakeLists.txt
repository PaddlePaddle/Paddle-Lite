set(dir ${CMAKE_CURRENT_SOURCE_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${dir}/build")
set(FOUND_MATCH OFF)
set(ENABLE_ALL_TEST ON)

if (ANDROID_ABI STREQUAL "arm64-v8a")
    message("using google's linker to link armv8 binary")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fuse-ld=gold")
endif ()

set(CON -1)

message(STATUS "nets :${NET}")

list(FIND NET "net" CON)
if (CON GREATER -1)
    # gen test
    ADD_EXECUTABLE(test-net net/test_net.cpp test_helper.h test_include.h executor_for_test.h)
    target_link_libraries(test-net paddle-mobile)
    set(FOUND_MATCH ON)
endif ()

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

    # gen test
    ADD_EXECUTABLE(test-mobilenetgpu net/test_mobilenet_GPU.cpp test_helper.h test_include.h)
    target_link_libraries(test-mobilenetgpu paddle-mobile)

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
    #ADD_EXECUTABLE(test-resnet50 fpga/test_resnet50.cpp test_helper.h test_include.h executor_for_test.h)
    #target_link_libraries(test-resnet50 paddle-mobile)

    #ADD_EXECUTABLE(test-densebox fpga/test_densebox_combine.cpp test_helper.h test_include.h executor_for_test.h)
    #target_link_libraries(test-densebox paddle-mobile)

    #ADD_EXECUTABLE(test-rfcn fpga/test_rfcn.cpp test_helper.h test_include.h executor_for_test.h)
    #target_link_libraries(test-rfcn paddle-mobile)

    #ADD_EXECUTABLE(test-marker fpga/test_marker.cpp test_helper.h test_include.h executor_for_test.h)
    #target_link_libraries(test-marker paddle-mobile)

    ADD_EXECUTABLE(test-rfcn-api fpga/test_rfcn_api.cpp)
    target_link_libraries(test-rfcn-api paddle-mobile)

    ADD_EXECUTABLE(test-mobilenet-api fpga/test_mobilenet_api.cpp)
    target_link_libraries(test-mobilenet-api paddle-mobile)

    ADD_EXECUTABLE(test-yolo-api fpga/test_yolo_api.cpp)
    target_link_libraries(test-yolo-api paddle-mobile)

    ADD_EXECUTABLE(test-marker-api fpga/test_marker_api.cpp)
    target_link_libraries(test-marker-api paddle-mobile)

    #ADD_EXECUTABLE(test-marker2 fpga/test_marker2.cpp test_helper.h test_include.h executor_for_test.h )
    #target_link_libraries(test-marker2 paddle-mobile)

    #ADD_EXECUTABLE(test-mobilenet fpga/test_mobilenet_beijing.cpp test_helper.h test_include.h executor_for_test.h)
    #target_link_libraries(test-mobilenet paddle-mobile)

    #ADD_EXECUTABLE(test-yolo fpga/test_yolo_combine.cpp test_helper.h test_include.h executor_for_test.h)
    #target_link_libraries(test-yolo paddle-mobile)

    set(FOUND_MATCH ON)
endif ()

list(FIND NET "FPGA_NET_V2" CON)
if (CON GREATER -1)
    ADD_EXECUTABLE(test-resnet50 fpga/test_resnet50.cpp test_helper.h test_include.h executor_for_test.h)
    target_link_libraries(test-resnet50 paddle-mobile)

    ADD_EXECUTABLE(test-pe fpga/test_pe.cpp)
    target_link_libraries(test-pe paddle-mobile)

    ADD_EXECUTABLE(test-densebox fpga/test_densebox_combine.cpp test_helper.h test_include.h executor_for_test.h)
    target_link_libraries(test-densebox paddle-mobile)

    set(FOUND_MATCH ON)
endif ()

list(FIND NET "FPGA_OPS_KD" CON)
if (CON GREATER -1)
    ADD_EXECUTABLE(test-ssd fpga/test_ssd.cpp test_helper.h test_include.h executor_for_test.h)
    target_link_libraries(test-ssd paddle-mobile)

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

list(FIND NET "super" CON)
if (CON GREATER -1)
    # gen test
    ADD_EXECUTABLE(test-super net/test_super.cpp test_helper.h test_include.h executor_for_test.h)
    target_link_libraries(test-super paddle-mobile)
    set(FOUND_MATCH ON)

endif ()

list(FIND NET "op" CON)
if (CON GREATER -1)
    #    # gen test
    #    ADD_EXECUTABLE(test-sigmoid operators/test_sigmoid_op.cpp test_include.h)
    #    target_link_libraries(test-sigmoid paddle-mobile)
    #
    #    # gen test log
    #    ADD_EXECUTABLE(test-leakyrelu operators/test_leaky_relu_op.cpp)
    #    target_link_libraries(test-leakyrelu paddle-mobile)
    # gen test log
    ADD_EXECUTABLE(test-log common/test_log.cpp)
    target_link_libraries(test-log paddle-mobile)
    set(FOUND_MATCH ON)
endif ()

if (ENABLE_ALL_TEST)
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
        ADD_EXECUTABLE(test-op-in-net net/test_op_in_net.cpp test_helper.h test_include.h executor_for_test.h)
        target_link_libraries(test-op-in-net paddle-mobile)

        # gen test
        ADD_EXECUTABLE(test-googlenet net/test_googlenet.cpp test_helper.h test_include.h executor_for_test.h)
        target_link_libraries(test-googlenet paddle-mobile)

        # gen test
        ADD_EXECUTABLE(test-googlenet-quali net/test_googlenet_quali.cpp test_helper.h test_include.h executor_for_test.h)
        target_link_libraries(test-googlenet-quali paddle-mobile)

        # gen test
        ADD_EXECUTABLE(test-conv-op operators/test_conv_op.cpp test_helper.h test_include.h executor_for_test.h)
        target_link_libraries(test-conv-op paddle-mobile)

        # gen test
        ADD_EXECUTABLE(test-expend-op operators/test_expend_op.cpp test_helper.h test_include.h executor_for_test_opencl.h)
        target_link_libraries(test-expend-op paddle-mobile)

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

        ADD_EXECUTABLE(test-relu6-op operators/test_relu6_op.cpp test_helper.h test_include.h)
        target_link_libraries(test-relu6-op paddle-mobile)

        ADD_EXECUTABLE(test-tanh-op operators/test_tanh_op.cpp test_helper.h test_include.h)
        target_link_libraries(test-tanh-op paddle-mobile)

        ADD_EXECUTABLE(test-log-op operators/test_log_op.cpp test_helper.h test_include.h)
        target_link_libraries(test-log-op paddle-mobile)

        ADD_EXECUTABLE(test-topk-op operators/test_topk_op.cpp test_helper.h test_include.h)
        target_link_libraries(test-topk-op paddle-mobile)

        ADD_EXECUTABLE(test-cast-op operators/test_cast_op.cpp test_helper.h test_include.h)
        target_link_libraries(test-cast-op paddle-mobile)

        ADD_EXECUTABLE(test-less-than-op operators/test_less_than_op.cpp test_helper.h test_include.h)
        target_link_libraries(test-less-than-op paddle-mobile)

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

        # gen test
        ADD_EXECUTABLE(test-optimize framework/test_optimize.cpp)
        target_link_libraries(test-optimize paddle-mobile)

        #gen test
        ADD_EXECUTABLE(test-pool-op operators/test_pool_op.cpp test_helper.h test_include.h executor_for_test.h)
        target_link_libraries(test-pool-op paddle-mobile)

        #gen test
        ADD_EXECUTABLE(test-softmax-op operators/test_softmax_op.cpp test_helper.h test_include.h executor_for_test.h)
        target_link_libraries(test-softmax-op paddle-mobile)

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
        ADD_EXECUTABLE(test-sigmoid-op operators/test_sigmoid_op.cpp test_include.h)
        target_link_libraries(test-sigmoid-op paddle-mobile)

        # gen test log
        ADD_EXECUTABLE(test-leakyrelu operators/test_leaky_relu_op.cpp)
        target_link_libraries(test-leakyrelu paddle-mobile)

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
        ADD_EXECUTABLE(test-mobilenetgpu net/test_mobilenet_GPU.cpp test_helper.h test_include.h)
        target_link_libraries(test-mobilenetgpu paddle-mobile)

        # gen test
        ADD_EXECUTABLE(test-yologpu net/test_yologpu.cpp test_helper.h test_include.h executor_for_test.h)
        target_link_libraries(test-yologpu paddle-mobile)

        # gen test
        ADD_EXECUTABLE(test-multi-process net/test_multi_inference_predict.cpp test_helper.h test_include.h)
        target_link_libraries(test-multi-process paddle-mobile)

        # gen test benchmark
        ADD_EXECUTABLE(test-benchmark net/test_benchmark.cpp)
        target_link_libraries(test-benchmark paddle-mobile)

        # gen test
        ADD_EXECUTABLE(test-eng net/test_eng.cpp test_helper.h test_include.h)
        target_link_libraries(test-eng paddle-mobile)

        # gen test
        ADD_EXECUTABLE(test-super net/test_super.cpp test_helper.h test_include.h)
        target_link_libraries(test-super paddle-mobile)

        # gen test
        ADD_EXECUTABLE(test-ocr net/test_ocr.cpp test_helper.h test_include.h)
        target_link_libraries(test-ocr paddle-mobile)

        ADD_EXECUTABLE(test-gesture net/test_gesture.cpp test_helper.h test_include.h)
        target_link_libraries(test-gesture paddle-mobile)

        ADD_EXECUTABLE(test-sequence-expand-op operators/test_sequence_expand_op.cpp test_helper.h test_include.h)
        target_link_libraries(test-sequence-expand-op paddle-mobile)

        ADD_EXECUTABLE(test-sequence-pool-op operators/test_sequence_pool_op.cpp test_helper.h test_include.h)
        target_link_libraries(test-sequence-pool-op paddle-mobile)

        ADD_EXECUTABLE(test-sequence-softmax-op operators/test_sequence_softmax_op.cpp test_helper.h test_include.h)
        target_link_libraries(test-sequence-softmax-op paddle-mobile)

        # gen test
        ADD_EXECUTABLE(test-vgg16ssd net/test_vgg16ssd.cpp test_helper.h test_include.h)
        target_link_libraries(test-vgg16ssd paddle-mobile)

        # gen test
        ADD_EXECUTABLE(test-logical-and-op operators/test_logical_and_op.cpp test_helper.h test_include.h)
        target_link_libraries(test-logical-and-op paddle-mobile)

        # gen test
        ADD_EXECUTABLE(test-logical-or-op operators/test_logical_or_op.cpp test_helper.h test_include.h)
        target_link_libraries(test-logical-or-op paddle-mobile)

        # gen test
        ADD_EXECUTABLE(test-logical-not-op operators/test_logical_not_op.cpp test_helper.h test_include.h)
        target_link_libraries(test-logical-not-op paddle-mobile)

        # gen test
        ADD_EXECUTABLE(test-logical-xor-op operators/test_logical_xor_op.cpp test_helper.h test_include.h)
        target_link_libraries(test-logical-xor-op paddle-mobile)

        # gen test
        ADD_EXECUTABLE(test-increment-op operators/test_increment_op.cpp test_helper.h test_include.h)
        target_link_libraries(test-increment-op paddle-mobile)

        # gen test
        ADD_EXECUTABLE(test-is-empty-op operators/test_is_empty_op.cpp test_helper.h test_include.h)
        target_link_libraries(test-is-empty-op paddle-mobile)

        ADD_EXECUTABLE(test-conv-bn-relu-op operators/test_conv_bn_relu_op.cpp test_helper.h test_include.h)
        target_link_libraries(test-conv-bn-relu-op paddle-mobile)

        ADD_EXECUTABLE(test-dwconv-bn-relu-op operators/test_dwconv_bn_relu_op.cpp test_helper.h test_include.h)
        target_link_libraries(test-dwconv-bn-relu-op paddle-mobile)

        ADD_EXECUTABLE(test-conv-gpu operators/test_conv_gpu.cpp test_helper.h test_include.h)
        target_link_libraries(test-conv-gpu paddle-mobile)

        ADD_EXECUTABLE(test-net-benchmark net/test_net_benchmark.cpp test_helper.h test_include.h)
        target_link_libraries(test-net-benchmark paddle-mobile)

        # gen test
        ADD_EXECUTABLE(test-net net/test_net.cpp test_helper.h test_include.h executor_for_test.h)
        target_link_libraries(test-net paddle-mobile)

        # gen test
        ADD_EXECUTABLE(test-net-feeds net/test_net_multi_feed.cpp test_helper.h test_include.h executor_for_test.h)
        target_link_libraries(test-net-feeds paddle-mobile)

        # gen test
        ADD_EXECUTABLE(test-net-performance net/test_net_performance.cpp test_helper.h test_include.h executor_for_test.h)
        target_link_libraries(test-net-performance paddle-mobile)

        ADD_EXECUTABLE(test-infer-imfix net/test_inference_imfix.cpp test_helper.h test_include.h executor_for_test.h)
        target_link_libraries(test-infer-imfix paddle-mobile)

#        ADD_EXECUTABLE(test-inference-ercy net/test_inference_ercy.cpp test_helper.h test_include.h executor_for_test.h)
#        target_link_libraries(test-inference-api-v2 paddle-mobile)

        if (GPU_CL)
            ADD_EXECUTABLE(test-net-male2fe net/test_mobilenet_male2fe.cpp test_helper.h test_include.h executor_for_test.h)
            target_link_libraries(test-net-male2fe paddle-mobile)

            ADD_EXECUTABLE(test-infer-m2fm net/test_inference_m2fm.cpp test_helper.h test_include.h executor_for_test.h)
            target_link_libraries(test-infer-m2fm  paddle-mobile)

        endif()

    endif ()
else ()
    # gen test
    ADD_EXECUTABLE(test-net net/test_net.cpp test_helper.h test_include.h executor_for_test.h)
    target_link_libraries(test-net paddle-mobile)

    ADD_EXECUTABLE(test-net-benchmark net/test_net_benchmark.cpp test_helper.h test_include.h)
    target_link_libraries(test-net-benchmark paddle-mobile)

#    ADD_EXECUTABLE(test-inference-ercy net/test_inference_ercy.cpp test_helper.h test_include.h executor_for_test.h)
#    target_link_libraries(test-inference-api-v2 paddle-mobile)
endif ()
