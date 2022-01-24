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

if(LITE_WITH_INTEL_FPGA)
  include(backends/intel_fpga)
endif()

if(LITE_WITH_NPU)
  include(backends/npu)
endif()

if(LITE_WITH_XPU)
  include(backends/xpu)
endif()

if(LITE_WITH_MLU)
  include(mlu)
endif()

if(LITE_WITH_CUDA)
  include(cuda)
endif()

if(LITE_WITH_BM)
  include(bm)
endif()

include(backends/x86)

# Add dependencies
include(generic)                # simplify cmake module
include(configure)              # add paddle env configuration
if(LITE_WITH_ARM)
  message(STATUS "Building the mobile framework")
  include(postproject)
  if(NOT LITE_ON_TINY_PUBLISH)
    include(external/gflags)    # download, build, install gflags
    include(external/gtest)     # download, build, install gtest
    include(ccache)
    include(external/protobuf)  # download, build, install protobuf
  endif()
else()
  include(coveralls)
  include(external/gflags)      # download, build, install gflags
  include(external/glog)        # download, build, install glog
  include(external/gtest)       # download, build, install gtest
  include(external/protobuf)    # download, build, install protobuf
  include(external/openblas)    # download, build, install openblas
  include(external/eigen)       # download eigen3
  include(cudnn)
  include(ccache)               # set ccache for compilation
  include(util)                 # set unittest and link libs
  include(version)              # set PADDLE_VERSION
  if(NOT APPLE)
    include(flags)
  endif()
  set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g -DNDEBUG")
  set(CMAKE_C_FLAGS_RELWITHDEBINFO "-O3 -g -DNDEBUG")
endif()
