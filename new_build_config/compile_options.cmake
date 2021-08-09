if(WIN32)
    option(MSVC_STATIC_CRT "use static C Runtime library by default" ON)

    set(CMAKE_SUPPRESS_REGENERATION ON)
    set(CMAKE_STATIC_LIBRARY_PREFIX lib)
    add_definitions("/DGOOGLE_GLOG_DLL_DECL=")

    if(MSVC_STATIC_CRT)
      set(CMAKE_C_FLAGS_DEBUG   "${CMAKE_C_FLAGS_DEBUG} /bigobj /MTd")
      set(CMAKE_C_FLAGS_RELEASE  "${CMAKE_C_FLAGS_RELEASE} /bigobj /MT")
      set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG} /bigobj /MTd")
      set(CMAKE_CXX_FLAGS_RELEASE   "${CMAKE_CXX_FLAGS_RELEASE} /bigobj /MT")
    else(MSVC_STATIC_CRT)
      set(CMAKE_C_FLAGS_DEBUG   "${CMAKE_C_FLAGS_DEBUG} /bigobj /MDd")
      set(CMAKE_C_FLAGS_RELEASE  "${CMAKE_C_FLAGS_RELEASE} /bigobj /MD")
      set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG} /bigobj /MDd")
      set(CMAKE_CXX_FLAGS_RELEASE   "${CMAKE_CXX_FLAGS_RELEASE} /bigobj /MD")
    endif(MSVC_STATIC_CRT)

    add_compile_options(/wd4068 /wd4129 /wd4244 /wd4267 /wd4297 /wd4530 /wd4577 /wd4819 /wd4838)
    add_compile_options(/MP)
    message(STATUS "Using parallel compiling (/MP)")
    set(PADDLE_LINK_FLAGS "/IGNORE:4006 /IGNORE:4098 /IGNORE:4217 /IGNORE:4221")
    set(CMAKE_STATIC_LINKER_FLAGS  "${CMAKE_STATIC_LINKER_FLAGS} ${PADDLE_LINK_FLAGS}")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${PADDLE_LINK_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} ${PADDLE_LINK_FLAGS}")
endif()

set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g -DNDEBUG")
set(CMAKE_C_FLAGS_RELWITHDEBINFO "-O3 -g -DNDEBUG")