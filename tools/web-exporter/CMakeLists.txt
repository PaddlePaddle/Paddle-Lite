cmake_minimum_required(VERSION 3.6)

project(web-exporter)

set(CMAKE_CXX_STANDARD 11)

file(GLOB PADDLE_MOBILE_CPP_FILES
  "../../src/common/*.c"
  "../../src/common/*.cpp"
  "../../src/memory/*.cpp"
  "../../src/framework/*.c"
  "../../src/framework/*.cpp"
  "../../src/framework/program/*.cpp"
  "../../src/framework/program/program-optimize/*.cpp"
)
file(GLOB EXPORT_CPP_FILES "*.cpp")

add_executable(web-exporter ${PADDLE_MOBILE_CPP_FILES} ${EXPORT_CPP_FILES})
target_include_directories(web-exporter PRIVATE "../../src")
target_link_libraries(web-exporter)