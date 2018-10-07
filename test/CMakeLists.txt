set(dir ${CMAKE_CURRENT_SOURCE_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${dir}/build")

if ("googlenet" IN_LIST NET)
    # gen test
    ADD_EXECUTABLE(test-googlenet net/test_googlenet.cpp test_helper.h  test_include.h executor_for_test.h)
    target_link_libraries(test-googlenet paddle-mobile)
elseif ("mobilenet" IN_LIST NET)
    # gen test
    ADD_EXECUTABLE(test-mobilenet net/test_mobilenet.cpp test_helper.h  test_include.h executor_for_test.h)
    target_link_libraries(test-mobilenet paddle-mobile)

    # gen test
    ADD_EXECUTABLE(test-mobilenet-combine net/test_mobilenet_combine.cpp test_helper.h  test_include.h executor_for_test.h)
    target_link_libraries(test-mobilenet-combine paddle-mobile)

elseif ("yolo" IN_LIST NET)
    # gen test
    ADD_EXECUTABLE(test-yolo net/test_yolo.cpp test_helper.h  test_include.h executor_for_test.h)
    target_link_libraries(test-yolo paddle-mobile)
elseif ("squeezenet" IN_LIST NET)
    # gen test
    ADD_EXECUTABLE(test-squeezenet net/test_squeezenet.cpp test_helper.h  test_include.h executor_for_test.h)
    target_link_libraries(test-squeezenet paddle-mobile)
elseif("resnet" IN_LIST NET)
    # gen test
    ADD_EXECUTABLE(test-resnet net/test_resnet.cpp test_helper.h  test_include.h executor_for_test.h)
    target_link_libraries(test-resnet paddle-mobile)
elseif("FPGAnets" IN_LIST NET)
    ADD_EXECUTABLE(test-resnet net/test_resnet.cpp test_helper.h  test_include.h executor_for_test.h)
    target_link_libraries(test-resnet paddle-mobile)

    ADD_EXECUTABLE(test-tensor-quant fpga/test_tensor_quant.cpp test_helper.h  test_include.h executor_for_test.h)
    target_link_libraries(test-tensor-quant paddle-mobile)

    ADD_EXECUTABLE(test-fpga-concat-op fpga/test_concat_op.cpp test_helper.h  test_include.h)
    target_link_libraries(test-fpga-concat-op paddle-mobile)

    ADD_EXECUTABLE(test-format-data fpga/test_format_data.cpp test_helper.h  test_include.h)
    target_link_libraries(test-format-data paddle-mobile)
elseif("mobilenetssd" IN_LIST NET)
    # gen test
    ADD_EXECUTABLE(test-mobilenetssd net/test_mobilenet+ssd.cpp test_helper.h  test_include.h executor_for_test.h)
    target_link_libraries(test-mobilenetssd paddle-mobile)

elseif("nlp" IN_LIST NET)
    # gen test
    ADD_EXECUTABLE(test-nlp net/test_nlp.cpp test_helper.h  test_include.h executor_for_test.h)
    target_link_libraries(test-nlp paddle-mobile)

    # gen test
    ADD_EXECUTABLE(test-gru-op  operators/test_gru_op.cpp test_helper.h  test_include.h)
    target_link_libraries(test-gru-op paddle-mobile)
else ()
    # gen test
    ADD_EXECUTABLE(test-googlenet net/test_googlenet.cpp test_helper.h  test_include.h executor_for_test.h)
    target_link_libraries(test-googlenet paddle-mobile)
endif()
