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

if(LITE_WITH_OPENCL)
  include(backends/opencl)
endif()

if(LITE_WITH_RKNPU)
  include(backends/rknpu)
endif()

if(LITE_WITH_INTEL_FPGA)
  include(backends/intel_fpga)
endif()

if(LITE_WITH_IMAGINATION_NNA)
  include(backends/imagination_nna)
endif()

if(LITE_WITH_NPU)
  include(backends/npu)
endif()

if(LITE_WITH_XPU)
  include(backends/xpu)
endif()

if(LITE_WITH_APU)
  include(backends/apu)
endif()

if(LITE_WITH_HUAWEI_ASCEND_NPU)
  include(backends/huawei_ascend_npu)
endif()

if(LITE_WITH_MLU)
  include(backends/mlu)
endif()

if(LITE_WITH_CUDA)
  include(backends/cuda)
endif()

if(LITE_WITH_BM)
  include(backends/bm)
endif()

include(backends/x86)

# Add dependencies
include(util/generic)                # simplify cmake module
include(util/configure)              # add paddle env configuration
if(WITH_LITE AND LITE_WITH_LIGHT_WEIGHT_FRAMEWORK)
  message(STATUS "Building the mobile framework")
  include(util/postproject)
  if(NOT LITE_ON_TINY_PUBLISH)
    include(external/gflags)    # download, build, install gflags
    include(external/gtest)     # download, build, install gtest
    include(external/ccache)
    include(external/protobuf)  # download, build, install protobuf
  endif()
else()
  include(util/coveralls)
  include(external/gflags)      # download, build, install gflags
  include(external/glog)        # download, build, install glog
  include(external/gtest)       # download, build, install gtest
  include(external/protobuf)    # download, build, install protobuf
  include(external/openblas)    # download, build, install openblas
  include(external/eigen)       # download eigen3
  include(external/cudnn)
  include(external/ccache)               # set ccache for compilation
  include(util/version)         # set PADDLE_VERSION
  if(NOT APPLE)
    include(util/flags)
  endif()
  set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g -DNDEBUG")
  set(CMAKE_C_FLAGS_RELWITHDEBINFO "-O3 -g -DNDEBUG")
endif()
