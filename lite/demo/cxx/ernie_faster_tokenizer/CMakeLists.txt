cmake_minimum_required(VERSION 3.10)
project(ernie_faster_tokenizer_demo)
enable_language(C)
enable_language(CXX)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -llog")

set(SRC_PATH ${PROJECT_SOURCE_DIR}/ernie_faster_tokenizer.cc)
add_executable(ernie_faster_tokenizer ${SRC_PATH})

set(FASTER_TOKERN_PATH ${PROJECT_SOURCE_DIR}/faster_tokenizer)
target_include_directories(ernie_faster_tokenizer PUBLIC ${FASTER_TOKERN_PATH}/include)
target_include_directories(ernie_faster_tokenizer PUBLIC ${FASTER_TOKERN_PATH}/third_party/include)
target_link_libraries(ernie_faster_tokenizer ${FASTER_TOKERN_PATH}/lib/libcore_tokenizers.so)

set(LITE_PATH ${PROJECT_SOURCE_DIR}/lite/cxx)
target_include_directories(ernie_faster_tokenizer PUBLIC ${LITE_PATH}/include)
target_link_libraries(ernie_faster_tokenizer ${LITE_PATH}/lib/libpaddle_light_api_shared.so)
