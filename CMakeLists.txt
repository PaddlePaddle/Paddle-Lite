cmake_minimum_required(VERSION 2.8.7)
project(mobile-deep-learning)
include_directories(include /usr/local/include)


set(CMAKE_VERBOSE_MAKEFILE on)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY build)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY build)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY build)


if(IS_IOS)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fobjc-abi-version=2 -fobjc-arc -isysroot ${CMAKE_OSX_SYSROOT}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fobjc-abi-version=2 -fobjc-arc -std=gnu++11 -stdlib=libc++ -isysroot ${CMAKE_OSX_SYSROOT}")
endif(IS_IOS)

# Add all headers and source files.
file(GLOB_RECURSE MDL_CPP src/*.cpp third-party/*.cpp)
file(GLOB_RECURSE MDL_H *.h)
list(REMOVE_ITEM MDL_CPP main.cpp)
if(ANDROID)
else(ANDROID)
    list(REMOVE_ITEM MDL_H ${CMAKE_CURRENT_SOURCE_DIR}/include/mdl_jni.h)
    list(REMOVE_ITEM MDL_CPP ${CMAKE_CURRENT_SOURCE_DIR}/src/mdl_jni.cpp)
    list(REMOVE_ITEM MDL_CPP ${CMAKE_CURRENT_SOURCE_DIR}/include/math/neon_mathfun.h)
endif(ANDROID)

# include headers
include_directories(include/)
include_directories(third-party/)

# gen so
ADD_LIBRARY(mdl SHARED ${MDL_CPP} ${MDL_H} fluid/test/test_helper.h)

# gen static
ADD_LIBRARY(mdl-static STATIC ${MDL_CPP} ${MDL_H})

# gen test
if(IS_IOS)
else(IS_IOS)
    ADD_EXECUTABLE(mdlTest test/main.cpp)
    target_link_libraries(mdlTest mdl-static)

endif(IS_IOS)

if(IS_MAC)
    add_subdirectory(tools)
    add_definitions(-DMDL_MAC="true")
elseif(V7)
    if(IS_IOS)
        add_definitions(-DMDL_V7_iOS="true")
    else(IS_IOS)
        add_definitions(-DMDL_V7="true")
    endif(IS_IOS)
elseif(V8)
    add_definitions(-DMDL_V8="true")
endif(IS_MAC)
