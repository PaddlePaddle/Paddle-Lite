set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -mavx -mfma -Wno-write-strings -Wno-psabi")

set(PADDLE_RESOURCE_URL "http://paddle-inference-dist.bj.bcebos.com" CACHE STRING "inference download url")

function(cc_library TARGET_NAME)
  set(options STATIC static SHARED shared)
  set(oneValueArgs "")
  set(multiValueArgs SRCS DEPS)
  cmake_parse_arguments(cc_library "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  if(cc_library_SRCS)
    if(cc_library_SHARED OR cc_library_shared) # build *.so
      add_library(${TARGET_NAME} SHARED ${cc_library_SRCS})
    else()
      add_library(${TARGET_NAME} STATIC ${cc_library_SRCS})
    endif()

    if(cc_library_DEPS)
      # Don't need link libwarpctc.so
      target_link_libraries(${TARGET_NAME} ${cc_library_DEPS})
      add_dependencies(${TARGET_NAME} ${cc_library_DEPS})
    endif()

    # cpplint code style
    foreach(source_file ${cc_library_SRCS})
      string(REGEX REPLACE "\\.[^.]*$" "" source ${source_file})
      if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${source}.h)
        list(APPEND cc_library_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/${source}.h)
      endif()
    endforeach()
  else(cc_library_SRCS)
    if(cc_library_DEPS)
      merge_static_libs(${TARGET_NAME} ${cc_library_DEPS})
    else()
      message(FATAL_ERROR "Please specify source files or libraries in cc_library(${TARGET_NAME} ...).")
    endif()
  endif(cc_library_SRCS)

    target_link_libraries(${TARGET_NAME} ${isl_lib} ${ginac_lib} Threads::Threads)

endfunction(cc_library)


function(cc_test TARGET_NAME)
  #if(WITH_TESTING)
    set(options SERIAL)
    set(oneValueArgs "")
    set(multiValueArgs SRCS DEPS ARGS)
    cmake_parse_arguments(cc_test "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    add_executable(${TARGET_NAME} ${cc_test_SRCS})
    get_property(os_dependency_modules GLOBAL PROPERTY OS_DEPENDENCY_MODULES)
    target_link_libraries(${TARGET_NAME} ${cc_test_DEPS} ${os_dependency_modules} gflags glog Catch2::Catch2)
    add_dependencies(${TARGET_NAME} ${cc_test_DEPS} gflags glog)

    add_test(NAME ${TARGET_NAME}
      COMMAND ${TARGET_NAME} "${cc_test_ARGS}"
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    if (${cc_test_SERIAL})
      set_property(TEST ${TARGET_NAME} PROPERTY RUN_SERIAL 1)
    endif()
    # No unit test should exceed 10 minutes.
    set_tests_properties(${TARGET_NAME} PROPERTIES TIMEOUT 6000)
  #endif()
endfunction()

function(nv_library TARGET_NAME)
  if (WITH_CUDA)
    set(options STATIC static SHARED shared)
    set(oneValueArgs "")
    set(multiValueArgs SRCS DEPS)
    cmake_parse_arguments(nv_library "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    if(nv_library_SRCS)
      if (nv_library_SHARED OR nv_library_shared) # build *.so
        cuda_add_library(${TARGET_NAME} SHARED ${nv_library_SRCS})
      else()
        cuda_add_library(${TARGET_NAME} STATIC ${nv_library_SRCS})
      endif()
      if (nv_library_DEPS)
        add_dependencies(${TARGET_NAME} ${nv_library_DEPS})
        target_link_libraries(${TARGET_NAME} ${nv_library_DEPS})
      endif()
      # cpplint code style
      foreach(source_file ${nv_library_SRCS})
        string(REGEX REPLACE "\\.[^.]*$" "" source ${source_file})
        if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${source}.h)
          list(APPEND nv_library_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/${source}.h)
        endif()
      endforeach()
    else(nv_library_SRCS)
      if (nv_library_DEPS)
        merge_static_libs(${TARGET_NAME} ${nv_library_DEPS})
      else()
        message(FATAL "Please specify source file or library in nv_library.")
      endif()
    endif(nv_library_SRCS)
    target_link_libraries(${TARGET_NAME} Threads::Threads)
  endif()
endfunction(nv_library)

function(nv_binary TARGET_NAME)
  if (WITH_CUDA)
    set(options "")
    set(oneValueArgs "")
    set(multiValueArgs SRCS DEPS)
    cmake_parse_arguments(nv_binary "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    cuda_add_executable(${TARGET_NAME} ${nv_binary_SRCS})
    if(nv_binary_DEPS)
      target_link_libraries(${TARGET_NAME} ${nv_binary_DEPS})
      add_dependencies(${TARGET_NAME} ${nv_binary_DEPS})
      common_link(${TARGET_NAME})
    endif()
  endif()
endfunction(nv_binary)

function(nv_test TARGET_NAME)
  if (WITH_CUDA AND WITH_TESTING)
    set(options SERIAL)
    set(oneValueArgs "")
    set(multiValueArgs SRCS DEPS ARGS)
    cmake_parse_arguments(nv_test "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    cuda_add_executable(${TARGET_NAME} ${nv_test_SRCS})
    get_property(os_dependency_modules GLOBAL PROPERTY OS_DEPENDENCY_MODULES)
    target_link_libraries(${TARGET_NAME} ${nv_test_DEPS} cinn_gtest_main gtest
gflags glog ${os_dependency_modules} ${CUDNN_LIBRARY} ${CUBLAS_LIBRARIES} ${CUDA_LIBRARIES})
    add_dependencies(${TARGET_NAME} ${nv_test_DEPS} cinn_gtest_main gtest gflags glog)
    common_link(${TARGET_NAME})
    # add_test(${TARGET_NAME} ${TARGET_NAME})
    add_test(NAME ${TARGET_NAME}
      COMMAND ${TARGET_NAME} "${nv_test_ARGS}"
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    if (nv_test_SERIAL)
        set_property(TEST ${TARGET_NAME} PROPERTY RUN_SERIAL 1)
    endif()
    target_link_libraries(${TARGET_NAME} Threads::Threads ${CUDA_NVRTC_LIB} ${CUDA_LIBRARIES} ${CUDA_cudart_static_LIBRARY}
      ${CUDA_TOOLKIT_ROOT_DIR}/lib64/stubs/libcuda.so
      )
  endif()
endfunction(nv_test)



# find all third_party modules is used for paddle static library
# for reduce the dependency when building the inference libs.
set_property(GLOBAL PROPERTY FLUID_THIRD_PARTY)
function(find_fluid_thirdparties TARGET_NAME)
  get_filename_component(__target_path ${TARGET_NAME} ABSOLUTE)
  string(REGEX REPLACE "^${PADDLE_SOURCE_DIR}/" "" __target_path ${__target_path})
  string(FIND "${__target_path}" "third_party" pos)
  if(pos GREATER 1)
    get_property(fluid_ GLOBAL PROPERTY FLUID_THIRD_PARTY)
    set(fluid_third_partys ${fluid_third_partys} ${TARGET_NAME})
    set_property(GLOBAL PROPERTY FLUID_THIRD_PARTY "${fluid_third_partys}")
  endif()
endfunction(find_fluid_thirdparties)

function(merge_static_libs TARGET_NAME)
  set(libs ${ARGN})
  list(REMOVE_DUPLICATES libs)

  # Get all propagation dependencies from the merged libraries
  foreach(lib ${libs})
    list(APPEND libs_deps ${${lib}_LIB_DEPENDS})
  endforeach()
  if(libs_deps)
    list(REMOVE_DUPLICATES libs_deps)
  endif()

  # To produce a library we need at least one source file.
  # It is created by add_custom_command below and will helps
  # also help to track dependencies.
  set(target_SRCS ${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME}_dummy.c)

  if(APPLE) # Use OSX's libtool to merge archives
    # Make the generated dummy source file depended on all static input
    # libs. If input lib changes,the source file is touched
    # which causes the desired effect (relink).
    add_custom_command(OUTPUT ${target_SRCS}
      COMMAND ${CMAKE_COMMAND} -E touch ${target_SRCS}
      DEPENDS ${libs})

    # Generate dummy staic lib
    file(WRITE ${target_SRCS} "const char *dummy_${TARGET_NAME} = \"${target_SRCS}\";")
    add_library(${TARGET_NAME} STATIC ${target_SRCS})
    target_link_libraries(${TARGET_NAME} ${libs_deps})

    foreach(lib ${libs})
      # Get the file names of the libraries to be merged
      set(libfiles ${libfiles} $<TARGET_FILE:${lib}>)
    endforeach()
    add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
      COMMAND rm "${CMAKE_CURRENT_BINARY_DIR}/lib${TARGET_NAME}.a"
      COMMAND /usr/bin/libtool -static -o "${CMAKE_CURRENT_BINARY_DIR}/lib${TARGET_NAME}.a" ${libfiles}
      )
  endif(APPLE)
  if(LINUX) # general UNIX: use "ar" to extract objects and re-add to a common lib
    set(target_DIR ${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME}.dir)

    foreach(lib ${libs})
      set(objlistfile ${target_DIR}/${lib}.objlist) # list of objects in the input library
      set(objdir ${target_DIR}/${lib}.objdir)

      add_custom_command(OUTPUT ${objdir}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${objdir}
        DEPENDS ${lib})

      add_custom_command(OUTPUT ${objlistfile}
        COMMAND ${CMAKE_AR} -x "$<TARGET_FILE:${lib}>"
        COMMAND ${CMAKE_AR} -t "$<TARGET_FILE:${lib}>" > ${objlistfile}
        DEPENDS ${lib} ${objdir}
        WORKING_DIRECTORY ${objdir})

      list(APPEND target_OBJS "${objlistfile}")
    endforeach()

    # Make the generated dummy source file depended on all static input
    # libs. If input lib changes,the source file is touched
    # which causes the desired effect (relink).
    add_custom_command(OUTPUT ${target_SRCS}
      COMMAND ${CMAKE_COMMAND} -E touch ${target_SRCS}
      DEPENDS ${libs} ${target_OBJS})

    # Generate dummy staic lib
    file(WRITE ${target_SRCS} "const char *dummy_${TARGET_NAME} = \"${target_SRCS}\";")
    add_library(${TARGET_NAME} STATIC ${target_SRCS})
    target_link_libraries(${TARGET_NAME} ${libs_deps})

    # Get the file name of the generated library
    set(target_LIBNAME "$<TARGET_FILE:${TARGET_NAME}>")

    add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
        COMMAND ${CMAKE_AR} crs ${target_LIBNAME} `find ${target_DIR} -name '*.o'`
        COMMAND ${CMAKE_RANLIB} ${target_LIBNAME}
        WORKING_DIRECTORY ${target_DIR})
  endif(LINUX)
  if(WIN32) # windows do not support gcc/nvcc combined compiling. Use msvc lib.exe to merge libs.
    # Make the generated dummy source file depended on all static input
    # libs. If input lib changes,the source file is touched
    # which causes the desired effect (relink).
    add_custom_command(OUTPUT ${target_SRCS}
      COMMAND ${CMAKE_COMMAND} -E touch ${target_SRCS}
      DEPENDS ${libs})

    # Generate dummy staic lib
    file(WRITE ${target_SRCS} "const char *dummy_${TARGET_NAME} = \"${target_SRCS}\";")
    add_library(${TARGET_NAME} STATIC ${target_SRCS})
    target_link_libraries(${TARGET_NAME} ${libs_deps})

    foreach(lib ${libs})
      # Get the file names of the libraries to be merged
      set(libfiles ${libfiles} $<TARGET_FILE:${lib}>)
    endforeach()
    # msvc will put libarary in directory of "/Release/xxxlib" by default
    #       COMMAND cmake -E remove "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_BUILD_TYPE}/${TARGET_NAME}.lib"
    add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
      COMMAND cmake -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_BUILD_TYPE}"
      COMMAND lib /OUT:${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_BUILD_TYPE}/lib${TARGET_NAME}.lib ${libfiles}
      )
  endif(WIN32)
endfunction(merge_static_libs)


# Modification of standard 'protobuf_generate_cpp()' with protobuf-lite support
# Usage:
#   paddle_protobuf_generate_cpp(<proto_srcs> <proto_hdrs> <proto_files>)

function(paddle_protobuf_generate_cpp SRCS HDRS)
  if(NOT ARGN)
    message(SEND_ERROR "Error: paddle_protobuf_generate_cpp() called without any proto files")
    return()
  endif()

  set(${SRCS})
  set(${HDRS})

  foreach(FIL ${ARGN})
    get_filename_component(ABS_FIL ${FIL} ABSOLUTE)
    get_filename_component(FIL_WE ${FIL} NAME_WE)

    set(_protobuf_protoc_src "${CMAKE_CURRENT_BINARY_DIR}/${FIL_WE}.pb.cc")
    set(_protobuf_protoc_hdr "${CMAKE_CURRENT_BINARY_DIR}/${FIL_WE}.pb.h")
    list(APPEND ${SRCS} "${_protobuf_protoc_src}")
    list(APPEND ${HDRS} "${_protobuf_protoc_hdr}")

    add_custom_command(
      OUTPUT "${_protobuf_protoc_src}"
             "${_protobuf_protoc_hdr}"

      COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_BINARY_DIR}"
      COMMAND ${PROTOBUF_PROTOC_EXECUTABLE}
      -I${CMAKE_CURRENT_SOURCE_DIR}
      --cpp_out "${CMAKE_CURRENT_BINARY_DIR}" ${ABS_FIL}
      DEPENDS ${ABS_FIL} protoc
      COMMENT "Running C++ protocol buffer compiler on ${FIL}"
      VERBATIM )
  endforeach()

  set_source_files_properties(${${SRCS}} ${${HDRS}} PROPERTIES GENERATED TRUE)
  set(${SRCS} ${${SRCS}} PARENT_SCOPE)
  set(${HDRS} ${${HDRS}} PARENT_SCOPE)
endfunction()


function(proto_library TARGET_NAME)
  set(oneValueArgs "")
  set(multiValueArgs SRCS DEPS)
  cmake_parse_arguments(proto_library "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  set(proto_srcs)
  set(proto_hdrs)
  paddle_protobuf_generate_cpp(proto_srcs proto_hdrs ${proto_library_SRCS})
  cc_library(${TARGET_NAME} SRCS ${proto_srcs} DEPS ${proto_library_DEPS} protobuf)
endfunction()

function(common_link TARGET_NAME)
  if (WITH_PROFILER)
    target_link_libraries(${TARGET_NAME} gperftools::profiler)
  endif()

  if (WITH_JEMALLOC)
    target_link_libraries(${TARGET_NAME} jemalloc::jemalloc)
  endif()
endfunction()


# This method is borrowed from Paddle-Lite.
function(download_and_uncompress INSTALL_DIR URL FILENAME)
  message(STATUS "Download inference test stuff from ${URL}/${FILENAME}")
  string(REGEX REPLACE "[-%.]" "_" FILENAME_EX ${FILENAME})
  set(EXTERNAL_PROJECT_NAME "extern_lite_download_${FILENAME_EX}")
  set(UNPACK_DIR "${INSTALL_DIR}/src/${EXTERNAL_PROJECT_NAME}")
  ExternalProject_Add(
    ${EXTERNAL_PROJECT_NAME}
    ${EXTERNAL_PROJECT_LOG_ARGS}
    PREFIX                ${INSTALL_DIR}
    DOWNLOAD_COMMAND      wget --no-check-certificate -q -O ${INSTALL_DIR}/${FILENAME} ${URL}/${FILENAME} && ${CMAKE_COMMAND} -E tar xzf ${INSTALL_DIR}/${FILENAME}
    DOWNLOAD_DIR          ${INSTALL_DIR}
    DOWNLOAD_NO_PROGRESS  1
    CONFIGURE_COMMAND     ""
    BUILD_COMMAND         ""
    UPDATE_COMMAND        ""
    INSTALL_COMMAND       ""
    )
endfunction()

# Add a source file to cinncore library.
# @param src_names: a list of strings
# usage:
# core_gather_srcs(SRCS a.cc b.cc c.cc)
function(core_gather_srcs)
  set(options)
  set(oneValueArgs)
  set(multiValueArgs "SRCS")
  cmake_parse_arguments(prefix "" "" "${multiValueArgs}" ${ARGN})
  foreach(cpp ${prefix_SRCS})
    set(core_src
      "${core_src};${CMAKE_CURRENT_SOURCE_DIR}/${cpp}"
      CACHE INTERNAL "")
  endforeach()

endfunction()

function(core_gather_headers)
  file(GLOB includes LIST_DIRECTORIES false RELATIVE ${CMAKE_SOURCE_DIR} *.h)

  foreach(header ${includes})
    set(core_includes "${core_includes};${header}" CACHE INTERNAL "")
  endforeach()
endfunction()
