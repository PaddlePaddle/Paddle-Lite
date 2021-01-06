if((NOT LITE_WITH_OPENCL AND NOT LITE_WITH_FPGA AND NOT LITE_WITH_MLU AND NOT LITE_WITH_XPU) AND (LITE_WITH_ARM))
    lite_cc_test(get_conv_latency SRCS src/get_conv_latency.cc DEPS arena_framework ${arm_kernels} ${lite_ops} ${host_kernels})
    lite_cc_test(get_batchnorm_latency SRCS src/get_batchnorm_latency.cc DEPS ${arm_kernels} ${lite_ops} ${host_kernels})
    lite_cc_test(get_pooling_latency SRCS src/get_pooling_latency.cc DEPS ${arm_kernels} ${lite_ops} ${host_kernels})
    lite_cc_test(get_fc_latency SRCS src/get_fc_latency.cc DEPS ${arm_kernels} ${lite_ops} ${host_kernels})
    lite_cc_test(get_activation_latency SRCS src/get_activation_latency.cc DEPS ${arm_kernels} ${lite_ops} ${host_kernels})
endif()

IF (LITE_WITH_BENCHMARK_TEST)
    # auto download google benchmark if necessary
    IF (NOT DEFINED GOOGLEBENCHMARK_SOURCE_DIR)
        SET(GOOGLEBENCHMARK_SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/googlebenchmark-source" CACHE STRING "Google Benchmark source directory")
        MESSAGE(STATUS "Auto downloading Google Benchmark , define GOOGLEBENCHMARK_SOURCE_DIR to the src repo to avoid auto downloading")

        CONFIGURE_FILE(${CMAKE_SOURCE_DIR}/cmake/DownloadGoogleBenchmark.cmake.in "${CMAKE_CURRENT_BINARY_DIR}/googlebenchmark-download/CMakeLists.txt")
        EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
                WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/googlebenchmark-download")
        EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" --build .
                WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/googlebenchmark-download")
    ENDIF ()

    # add google benchmark as subdirectory
    IF (CMAKE_SYSTEM_NAME MATCHES "^(Android|iOS)$")
        SET(HAVE_POSIX_REGEX CACHE BOOL 0)
        SET(HAVE_STEADY_CLOCK CACHE BOOL 0)
        SET(HAVE_STD_REGEX CACHE BOOL 0)
    ENDIF ()
    SET(BENCHMARK_ENABLE_GTEST_TESTS CACHE BOOL 0)
    SET(GOOGLEBENCHMARK_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/googlebenchmark-build" CACHE STRING "Google Benchmark build directory")
    ADD_SUBDIRECTORY(${GOOGLEBENCHMARK_SOURCE_DIR} ${GOOGLEBENCHMARK_BUILD_DIR})

    #add test cases
    if(LITE_WITH_ARM)
        lite_cc_test(f32-gemm-bench-arm SRCS src/f32-gemm-arm.cc DEPS ${arm_kernels} ${lite_ops} ${host_kernels} benchmark)
        lite_cc_test(elementwise-arm-math-bench SRCS src/elementwise_arm_math.cpp DEPS ${arm_kernels} ${lite_ops} ${host_kernels} benchmark)
        lite_cc_test(int8-gemm-bench-arm SRCS src/int8-gemm-arm.cc DEPS ${arm_kernels} ${lite_ops} ${host_kernels} benchmark)
        lite_cc_test(conv-bench-arm SRCS src/convolution-arm.cc DEPS ${arm_kernels} ${lite_ops} ${host_kernels} benchmark)
    endif()

ENDIF ()
