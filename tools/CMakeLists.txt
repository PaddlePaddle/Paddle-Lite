
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../src)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/../src)

find_package(Protobuf REQUIRED)

include_directories(${CMAKE_CURRENT_BINARY_DIR})
protobuf_generate_cpp(CAFFE_PROTO_SRCS CAFFE_PROTO_HDRS caffe.proto)

add_executable(caffe2mdl caffe2mdl.cpp ${CAFFE_PROTO_SRCS} ${CAFFE_PROTO_HDRS})

target_link_libraries(caffe2mdl ${PROTOBUF_LIBRARIES})

include_directories(${CMAKE_SOURCE_DIR}/src)

