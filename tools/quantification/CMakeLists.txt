set(dir ${CMAKE_CURRENT_SOURCE_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${dir}/build")

ADD_EXECUTABLE(convert convert.cpp)
target_link_libraries(convert paddle-mobile)