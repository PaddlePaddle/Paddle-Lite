#----------------------------------------------- NOT CHANGE ---------------------------------------
find_package(PythonInterp REQUIRED)
# A trick to generate the paddle_use_kernels.h
execute_process(
    COMMAND ${CMAKE_COMMAND} -E copy ${PADDLE_SOURCE_DIR}/lite/api/paddle_lite_factory_helper.h ${PADDLE_BINARY_DIR}/lite/api/)
execute_process(
    COMMAND ${PYTHON_EXECUTABLE} ${PADDLE_SOURCE_DIR}/lite/tools/cmake_tools/parse_kernel_registry.py
    ${kernels_src_list}
    ${fake_kernels_src_list}
    ${PADDLE_BINARY_DIR}/lite/api/paddle_use_kernels.h
    "${LITE_OPTMODEL_DIR}/.tailored_kernels_list"
    ${LITE_BUILD_TAILOR}
    ${LITE_BUILD_EXTRA}
    ${LITE_WITH_ARM82_FP16}
    RESULT_VARIABLE result
    ERROR_VARIABLE error)
if(NOT "${error}" STREQUAL "")
    message(FATAL_ERROR ${error})
endif()
# A trick to generate the paddle_use_ops.h
execute_process(
    COMMAND ${PYTHON_EXECUTABLE} ${PADDLE_SOURCE_DIR}/lite/tools/cmake_tools/parse_op_registry.py
    ${ops_src_list}
    ${PADDLE_BINARY_DIR}/lite/api/paddle_use_ops.h
    "${LITE_OPTMODEL_DIR}/.tailored_ops_list"
    ${LITE_BUILD_TAILOR}
    ${LITE_BUILD_EXTRA}
    RESULT_VARIABLE result
    ERROR_VARIABLE error)
if(NOT "${error}" STREQUAL "")
    message(FATAL_ERROR ${error})
endif()
#----------------------------------------------- NOT CHANGE ---------------------------------------

set(LIGHT_API_SRC  light_api.cc paddle_api.cc light_api_impl.cc paddle_place.cc)
set(FULL_API_SRC ${LIGHT_API_SRC} cxx_api.cc cxx_api_impl.cc)
set(light_lib_DEPS utils core kernels model_parser ops CACHE INTERNAL "")
set(full_lib_DEPS framework_proto core ops utils kernels model_parser CACHE INTERNAL "")
set(external_libs_DEPS "" CACHE INTERNAL "")

if (LITE_ON_TINY_PUBLISH)
    set(CMAKE_CXX_FLAGS_RELEASE "-Os -DNDEBUG")
    set(CMAKE_C_FLAGS_RELEASE "-Os -DNDEBUG")
endif()
if(MSVC)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /bigobj")
endif()


if(LITE_WITH_FPGA)
    set(external_libs_DEPS ${external_libs_DEPS} ${fpga_deps})
endif()
if(LITE_WITH_BM)
    set(external_libs_DEPS ${external_libs_DEPS} ${bm_deps})
endif()
if(LITE_WITH_INTEL_FPGA)
    set(external_libs_DEPS ${external_libs_DEPS} ${intel_fpga_deps})
endif()
if(LITE_WITH_NNADAPTER)
    set(external_libs_DEPS ${external_libs_DEPS} ${nnadapter_deps})
endif()
# for light api
if(LITE_WITH_CUDA)
    get_property(cuda_deps GLOBAL PROPERTY CUDA_MODULES)
    set(external_libs_DEPS ${external_libs_DEPS} target_wrapper_cuda)
endif()


lite_cc_library(paddle_api_light SRCS ${LIGHT_API_SRC} DEPS ${light_lib_DEPS} ${external_libs_DEPS})
if(NOT LITE_ON_TINY_PUBLISH)
  lite_cc_library(paddle_api_full SRCS ${FULL_API_SRC} DEPS ${full_lib_DEPS} ${external_libs_DEPS})
endif()

if (NOT LITE_ON_TINY_PUBLISH)
    #dynamic library
    lite_cc_library(paddle_light_api_shared SHARED SRCS ${LIGHT_API_SRC}
                  DEPS ${light_lib_DEPS} ${external_libs_DEPS} paddle_api_light)
    lite_cc_library(paddle_full_api_shared SHARED SRCS ${FULL_API_SRC}
                  DEPS ${full_lib_DEPS} ${external_libs_DEPS} paddle_api_full)
    add_dependencies(paddle_full_api_shared fbs_headers)
    if (LITE_WITH_CUDA)
        target_link_libraries(paddle_full_api_shared "-Wl,--whole-archive" ${math_cuda} "-Wl,--no-whole-archive")
    endif()
    if(LITE_WITH_METAL)
        target_link_libraries(paddle_full_api_shared ${METAL_LIBRARY} ${MPS_LIBRARY} ${GRAPHIC} ${FOUNDATION_LIBRARY})
    endif()


    #static library
    bundle_static_library(paddle_api_full paddle_api_full_bundled bundle_full_api)
    bundle_static_library(paddle_api_light paddle_api_light_bundled bundle_light_api)

    if(WIN32)
        target_link_libraries(paddle_full_api_shared shlwapi.lib)
        target_link_libraries(paddle_light_api_shared shlwapi.lib)
    endif()
    # strip useless symbols
    if(${HOST_SYSTEM} MATCHES "macosx" AND NOT (ARM_TARGET_OS STREQUAL "android"))
        set(LINK_MAP_FILE "${PADDLE_SOURCE_DIR}/lite/core/exported_symbols.lds")
        set(LINK_FLAGS "-Wl,-exported_symbols_list ${LINK_MAP_FILE}")
        add_custom_command(OUTPUT ${LINK_MAP_FILE} COMMAND :)
        add_custom_target(custom_linker_map DEPENDS ${LINK_MAP_FILE})
        set_target_properties(paddle_full_api_shared PROPERTIES LINK_FLAGS ${LINK_FLAGS})
        add_dependencies(paddle_full_api_shared custom_linker_map)
        set_target_properties(paddle_light_api_shared PROPERTIES LINK_FLAGS ${LINK_FLAGS})
        add_dependencies(paddle_light_api_shared custom_linker_map)
   elseif(NOT WIN32)
        set(LINK_MAP_FILE "${PADDLE_SOURCE_DIR}/lite/core/lite.map")
        set(LINK_FLAGS "-Wl,--version-script ${LINK_MAP_FILE}")
        add_custom_command(OUTPUT ${LINK_MAP_FILE} COMMAND :)
        add_custom_target(custom_linker_map DEPENDS ${LINK_MAP_FILE})
        set_target_properties(paddle_full_api_shared PROPERTIES LINK_FLAGS ${LINK_FLAGS})
        add_dependencies(paddle_full_api_shared custom_linker_map)
        set_target_properties(paddle_light_api_shared PROPERTIES LINK_FLAGS ${LINK_FLAGS})
        add_dependencies(paddle_light_api_shared custom_linker_map)
   endif()
   if(NOT WIN32 AND NOT ARM_TARGET_OS STREQUAL "android" AND WITH_TESTING)
        # check symbol hidden
        FILE(WRITE ${CMAKE_CURRENT_BINARY_DIR}/check_symbol.cmake
            "execute_process(COMMAND sh -c \"${CMAKE_CURRENT_SOURCE_DIR}/../tools/check_symbol.sh"
            " ${CMAKE_CURRENT_BINARY_DIR}/libpaddle_light_api_shared.so\" RESULT_VARIABLE symbol_res)\n"
            "if(NOT \"\${symbol_res}\" STREQUAL \"0\")\n"
            "  message(FATAL_ERROR \"Check libpaddle_light_api_shared.so symbol failed.\")\n"
            "endif()\n"
            "execute_process(COMMAND sh -c \"${CMAKE_CURRENT_SOURCE_DIR}/../tools/check_symbol.sh"
            " ${CMAKE_CURRENT_BINARY_DIR}/libpaddle_full_api_shared.so\" RESULT_VARIABLE symbol_res)\n"
            "if(NOT \"\${symbol_res}\" STREQUAL \"0\")\n"
            "  message(FATAL_ERROR \"Check libpaddle_full_api_shared.so symbol failed.\")\n"
            "endif()\n")
        add_custom_command(
            OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/.check_symbol"
            COMMAND ${CMAKE_COMMAND} -P "${CMAKE_CURRENT_BINARY_DIR}/check_symbol.cmake"
            DEPENDS paddle_light_api_shared paddle_full_api_shared )
        add_custom_target(check_symbol ALL DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/.check_symbol")
        add_dependencies(lite_compile_deps check_symbol)
    endif()
else()
    # Compiling steps in tiny_publish format:
    # 1. compile all source files into .object `PADDLELITE_OBJS`
    add_library(PADDLELITE_OBJS OBJECT ${__lite_cc_files})
    add_dependencies(PADDLELITE_OBJS fbs_headers)
    if (IOS)
        # only sttaic lib is produced for IOS platform.
        add_library(paddle_api_light_bundled STATIC ${__lite_cc_files})
        add_dependencies(paddle_api_light_bundled fbs_headers)
    else()
        # 1. enable -flto compiling flag if toochain==gcc
        # TODO (hong19860320): Disable lto temporarily since it causes fail to catch the exceptions in android when toolchain is gcc.
        if (NOT (ARM_TARGET_LANG STREQUAL "clang"))
            if (((ARM_TARGET_OS STREQUAL "android" OR ARM_TARGET_OS STREQUAL "armmacos") AND LITE_WITH_EXCEPTION) OR ARM_TARGET_OS STREQUAL "qnx")
                set(TARGET_COMIPILE_FLAGS "")
            else()
                set(TARGET_COMIPILE_FLAGS "-fdata-sections -flto")
            endif()
        else()
            if (NOT LITE_WITH_METAL)
                set(TARGET_COMIPILE_FLAGS "-fdata-sections")
            endif()
        endif()
        #   1.1 enable -flto on PADDLELITE_OBJS
        set_target_properties(PADDLELITE_OBJS PROPERTIES COMPILE_FLAGS "${TARGET_COMIPILE_FLAGS}")

        # 2. produce dynamic lib from PADDLELITE_OBJS
        add_library(paddle_light_api_shared SHARED $<TARGET_OBJECTS:PADDLELITE_OBJS>)
        set_target_properties(paddle_light_api_shared PROPERTIES COMPILE_FLAGS "${TARGET_COMIPILE_FLAGS}")
        #   2.1 link `paddle_light_api_shared` to third-party libs
        if (LITE_WITH_METAL)
            target_link_libraries(paddle_light_api_shared ${METAL_LIBRARY} ${MPS_LIBRARY} ${FOUNDATION_LIBRARY})
        endif()
        if (LITE_WITH_NPU)
            # Need to add HIAI runtime libs (libhiai.so) dependency
            target_link_libraries(paddle_light_api_shared ${npu_builder_libs} ${npu_runtime_libs})
        endif()

        # 3. produce java lib from `PADDLELITE_OBJS` if LITE_WITH_JAVA=ON
        if (LITE_WITH_JAVA)
          add_library(paddle_lite_jni SHARED $<TARGET_OBJECTS:PADDLELITE_OBJS> android/jni/native/paddle_lite_jni.cc android/jni/native/tensor_jni.cc)
          set_target_properties(paddle_lite_jni PROPERTIES COMPILE_FLAGS "${TARGET_COMIPILE_FLAGS}")
          set_target_properties(paddle_lite_jni PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PADDLE_BINARY_DIR}/lite/api/android/jni/native)
          if (LITE_WITH_NPU)
              # Need to add HIAI runtime libs (libhiai.so) dependency
              target_link_libraries(paddle_lite_jni ${npu_builder_libs} ${npu_runtime_libs})
          endif()
        endif()

        # 4. produce static lib `libpaddle_api_light_bundled.a` from `PADDLELITE_OBJS`
        # '-flto' is only supported by dynamic lib
        if(TARGET_COMIPILE_FLAGS MATCHES ".*-flto.*")
            add_library(paddle_api_light_bundled STATIC ${__lite_cc_files} ${LIGHT_API_SRC})
        else()
            add_library(paddle_api_light_bundled STATIC $<TARGET_OBJECTS:PADDLELITE_OBJS>)
        endif()
    endif()
endif()


#-----------------------------------------------------------------------------------------------------
if(WITH_TESTING)
    add_subdirectory(test)
endif()

# java API
if (LITE_WITH_JAVA AND LITE_WITH_ARM)
    add_subdirectory(android)
endif()
# python API
if(LITE_WITH_PYTHON)
    # add library for opt_base
    add_subdirectory(python)
    lite_cc_library(lite_pybind SHARED SRCS python/pybind/pybind.cc tools/opt_base.cc DEPS ${full_lib_DEPS} paddle_api_full ${external_libs_DEPS}  pybind python)
    if(LITE_WITH_METAL)
        target_link_libraries(lite_pybind ${METAL_LIBRARY} ${MPS_LIBRARY} ${GRAPHIC} ${FOUNDATION_LIBRARY})
    endif()
endif()

if(NOT IOS AND NOT LITE_ON_TINY_PUBLISH)
    # test_model_bin
    lite_cc_binary(test_model_bin SRCS tools/model_test.cc
        DEPS gflags
        CV_DEPS paddle_cv_arm)

    # benchmark_bin
    add_subdirectory(tools/benchmark)
endif()

# opt tool
if (LITE_ON_MODEL_OPTIMIZE_TOOL)
    message(STATUS "Compiling opt")
    lite_cc_binary(opt SRCS tools/opt.cc tools/opt_base.cc
        DEPS gflags)
    if (EMSCRIPTEN)
        target_link_libraries(opt protobuf)
        set_target_properties(opt PROPERTIES LINK_FLAGS "-s INITIAL_MEMORY=1000MB -s ALLOW_MEMORY_GROWTH=1 -sASSERTIONS")
        if (WITH_NODE_RAW_FS)
            set_target_properties(opt PROPERTIES LINK_FLAGS "-s NODERAWFS=1")
        endif()
    endif()
endif()
