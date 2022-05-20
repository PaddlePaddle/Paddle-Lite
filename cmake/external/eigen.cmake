INCLUDE(ExternalProject)

SET(EIGEN_SOURCECODE_DIR ${PADDLE_SOURCE_DIR}/third-party/eigen3)
SET(EIGEN_SOURCE_DIR ${THIRD_PARTY_PATH}/eigen3)
SET(EIGEN_INCLUDE_DIR ${EIGEN_SOURCE_DIR}/src/extern_eigen3)
INCLUDE_DIRECTORIES(${EIGEN_INCLUDE_DIR})
if(NOT WITH_FAST_MATH)
  # EIGEN_FAST_MATH: https://eigen.tuxfamily.org/dox/TopicPreprocessorDirectives.html
  # enables some optimizations which might affect the accuracy of the result.
  # This currently enables the SSE vectorization of sin() and cos(),
  # and speedups sqrt() for single precision.
  # Defined to 1 by default. Define it to 0 to disable.
  add_definitions(-DEIGEN_FAST_MATH=0)
endif()

if(WITH_AMD_GPU)
    ExternalProject_Add(
        extern_eigen3
        ${EXTERNAL_PROJECT_LOG_ARGS}
        GIT_TAG
        URL             http://paddle-inference-dist.bj.bcebos.com/PaddleLite_ThirdParty%2Fhipeigen-upstream-702834151eaebcf955fd09ed0ad83c06.zip
        DOWNLOAD_DIR          ${EIGEN_SOURCECODE_DIR}
        DOWNLOAD_NO_PROGRESS  1
        PREFIX          ${EIGEN_SOURCE_DIR}
        DOWNLOAD_NAME   "hipeigen-upstream-702834151eaebcf955fd09ed0ad83c06.zip"
        UPDATE_COMMAND  ""
        CONFIGURE_COMMAND ""
        BUILD_COMMAND     ""
        INSTALL_COMMAND   ""
        TEST_COMMAND      ""
    )
else()
    ExternalProject_Add(
        extern_eigen3
        ${EXTERNAL_PROJECT_LOG_ARGS}
        # eigen on cuda9.1 missing header of math_funtions.hpp
        # https://stackoverflow.com/questions/43113508/math-functions-hpp-not-found-when-using-cuda-with-eigen
        GIT_TAG
        ######################################################################################################
        # url address of eigen before v2.3.0
        # URL             http://paddle-inference-dist.bj.bcebos.com/PaddleLite_ThirdParty%2Feigen-git-mirror-master-9ab917e9db99f5907d086aa73d5f9103.zip
        ######################################################################################################
        # url address of eigen since  v2.6.0
        #         github address: https://github.com/eigenteam/eigen-git-mirror
        # we changed the source code to adapt for windows compiling
        #         git diffs : (1) unsupported/Eigen/CXX11/src/Tensor/TensorBlockV2.h
        ######################################################################################################
        URL             http://paddlelite-data.bj.bcebos.com/third_party_libs/eigen-git-mirror-master-9ab917e9db99f5907d086aa73d5f9103.zip
        DOWNLOAD_DIR          ${EIGEN_SOURCECODE_DIR}
        DOWNLOAD_NO_PROGRESS  1
        PREFIX          ${EIGEN_SOURCE_DIR}
        DOWNLOAD_NAME   "eigen-git-mirror-master-9ab917e9db99f5907d086aa73d5f9103.zip"
        UPDATE_COMMAND  ""
        CONFIGURE_COMMAND ""
        BUILD_COMMAND     ""
        INSTALL_COMMAND   ""
        TEST_COMMAND      ""
    )
endif()

if (${CMAKE_VERSION} VERSION_LESS "3.3.0")
    set(dummyfile ${CMAKE_CURRENT_BINARY_DIR}/eigen3_dummy.c)
    file(WRITE ${dummyfile} "const char *dummy_eigen3 = \"${dummyfile}\";")
    add_library(eigen3 STATIC ${dummyfile})
else()
    add_library(eigen3 INTERFACE)
endif()

add_dependencies(eigen3 extern_eigen3)
