# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# for mobile
if (WITH_LITE AND LITE_WITH_LIGHT_WEIGHT_FRAMEWORK)
    message(STATUS "Building the mobile framework")
    include(postproject)
    include(backends/npu) # check and prepare NPU DDK
    include(backends/xpu) # check and prepare XPU SDK
    include(backends/apu) # check and prepare APU SDK 
    include(backends/huawei_ascend_npu)  # check and prepare Ascend NPU SDK 

    # We compile the mobile deployment library when LITE_ON_TINY_PUBLISH=ON
    # So the following third party dependencies are not needed.
    if (NOT LITE_ON_TINY_PUBLISH)
        # include the necessary thirdparty dependencies
        include(external/gflags)    # download, build, install gflags
        # LITE_WITH_LIGHT_WEIGHT_FRAMEWORK=ON will disable glog
        # TODO(sangoly): refine WITH_LITE and LITE_WITH_LIGHT_WEIGHT_FRAMEWORK
        include(external/gtest)     # download, build, install gtest
        include(ccache)             # set ccache for compilation
        include(external/protobuf)  # download, build, install protobuf
    endif()

    include(generic)            # simplify cmake module
    include(configure)          # add paddle env configuration

    add_subdirectory(lite)
    return()
endif()
#################################  End of mobile compile ##############################

set(WITH_MKLML ${WITH_MKL})
if (NOT DEFINED WITH_MKLDNN)
    if (WITH_MKL AND AVX2_FOUND)
        set(WITH_MKLDNN ON)
    else()
        message(STATUS "Do not have AVX2 intrinsics and disabled MKL-DNN")
        set(WITH_MKLDNN OFF)
    endif()
endif()

########################################################################################

if(LITE_WITH_XPU)
    include(backends/xpu)
endif()

if(LITE_WITH_MLU)
    include(mlu)
endif()

if(LITE_WITH_HUAWEI_ASCEND_NPU)
    include(backends/huawei_ascend_npu)
endif()

include(coveralls)

include(external/mklml)     # download mklml package
include(external/xbyak)     # download xbyak package

include(external/libxsmm)   # download, build, install libxsmm
include(external/gflags)    # download, build, install gflags
include(external/glog)      # download, build, install glog
include(external/gtest)     # download, build, install gtest
include(external/protobuf)  # download, build, install protobuf
include(external/openblas)  # download, build, install openblas
include(external/mkldnn)    # download, build, install mkldnn
include(external/eigen)     # download eigen3
include(external/xxhash)    # download install xxhash needed for x86 jit

include(cudnn)
include(configure)          # add paddle env configuration

if(LITE_WITH_CUDA)
  include(cuda)
endif()

if(LITE_WITH_BM)
  include(bm)
endif()
include(generic)            # simplify cmake module
include(ccache)             # set ccache for compilation
include(util)               # set unittest and link libs
include(version)            # set PADDLE_VERSION
if(NOT APPLE)
  include(flags)
endif()

set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g -DNDEBUG")
set(CMAKE_C_FLAGS_RELWITHDEBINFO "-O3 -g -DNDEBUG")


if(APPLE)
    if(NOT DEFINED ENABLE_ARC)
        # Unless specified, enable ARC support by default
        set(ENABLE_ARC TRUE)
        message(STATUS "Enabling ARC support by default. ENABLE_ARC not provided!")
    endif()

    set(ENABLE_ARC_INT ${ENABLE_ARC} CACHE BOOL "Whether or not to enable ARC" ${FORCE_CACHE})
    if(ENABLE_ARC_INT)
        set(FOBJC_ARC "-fobjc-arc")
        set(CMAKE_XCODE_ATTRIBUTE_CLANG_ENABLE_OBJC_ARC YES CACHE INTERNAL "")
        message(STATUS "Enabling ARC support.")
    else()
        set(FOBJC_ARC "-fno-objc-arc")
        set(CMAKE_XCODE_ATTRIBUTE_CLANG_ENABLE_OBJC_ARC NO CACHE INTERNAL "")
        message(STATUS "Disabling ARC support.")
    endif()
    set(CMAKE_C_FLAGS "${FOBJC_ARC} ${CMAKE_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${FOBJC_ARC} ${CMAKE_CXX_FLAGS}")
endif ()
add_subdirectory(lite)
