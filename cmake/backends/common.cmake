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

# for opencl
if (LITE_WITH_OPENCL)
  include_directories("${PADDLE_SOURCE_DIR}/third-party/opencl/include")
endif()

if(LITE_WITH_RKNPU)
  include(backends/rknpu)
endif()

if(LITE_WITH_INTEL_FPGA)
  include(backends/intel_fpga)
endif()

# for mobile
if (WITH_LITE AND LITE_WITH_LIGHT_WEIGHT_FRAMEWORK)
  message(STATUS "Building the mobile framework")
  include(postproject)
  include(backends/npu) # check and prepare NPU DDK
  include(backends/xpu) # check and prepare XPU
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

  return()
endif()
#################################  End of mobile compile ##############################

#################################  Server compile  ##############################

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
