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
  include(backend/opencl)
endif()

if(LITE_WITH_RKNPU)
  include(backend/rknpu)
endif()

if(LITE_WITH_INTEL_FPGA)
  include(backend/intel_fpga)
endif()

if(LITE_WITH_NPU)
  include(backend/npu)
endif()

if(LITE_WITH_XPU)
  include(backend/xpu)
endif()

if(LITE_WITH_MLU)
  include(backend/mlu)
endif()

if(LITE_WITH_CUDA)
  include(backend/cuda)
endif()

if(LITE_WITH_BM)
  include(backend/bm)
endif()

include(backend/x86)

# Add dependencies
include(config/generic)                # simplify cmake module
include(config/configure)              # add paddle env configuration
if(LITE_WITH_ARM)
  message(STATUS "Building the mobile framework")
  include(config/postproject)
  if(NOT LITE_ON_TINY_PUBLISH)
    include(external/gflags)    # download, build, install gflags
    include(external/gtest)     # download, build, install gtest
    include(external/ccache)
    include(external/protobuf)  # download, build, install protobuf
  endif()
else()
  include(external/coveralls)
  include(external/gflags)      # download, build, install gflags
  include(external/glog)        # download, build, install glog
  include(external/gtest)       # download, build, install gtest
  include(external/protobuf)    # download, build, install protobuf
  include(external/openblas)    # download, build, install openblas
  include(external/eigen)       # download eigen3
  include(external/cudnn)
  include(external/ccache)               # set ccache for compilation
  include(util)                 # set unittest and link libs
  include(config/version)              # set PADDLE_VERSION
  if(NOT APPLE)
    include(config/flags)
  endif()
  set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g -DNDEBUG")
  set(CMAKE_C_FLAGS_RELWITHDEBINFO "-O3 -g -DNDEBUG")
endif()
