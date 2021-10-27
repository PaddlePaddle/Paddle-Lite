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

# Backends
# Rockchip NPU
if(LITE_WITH_RKNPU)
  include(backends/rknpu)
endif()
# Intel FPGA
if(LITE_WITH_INTEL_FPGA)
  include(backends/intel_fpga)
endif()
# Baidu XPU
if(LITE_WITH_XPU)
  include(backends/xpu)
endif()
# MLU
if(LITE_WITH_MLU)
  include(mlu)
endif()
# BM
if(LITE_WITH_BM)
  include(bm)
endif()
# Huawei Kirin NPU
if(LITE_WITH_NPU)
  include(backends/npu)
endif()

include(backends/gpu)
include(backends/cpu)

# Backends dependences
if(WITH_LITE AND LITE_WITH_LIGHT_WEIGHT_FRAMEWORK)
  include(postproject)
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

else()

  include(coveralls)
  include(external/gflags)    # download, build, install gflags
  include(external/glog)      # download, build, install glog
  include(external/gtest)     # download, build, install gtest
  include(external/protobuf)  # download, build, install protobuf
  include(configure)          # add paddle env configuration

  include(generic)            # simplify cmake module
  include(ccache)             # set ccache for compilation
  include(util)               # set unittest and link libs
  include(version)            # set PADDLE_VERSION
  if(NOT APPLE)
    include(flags)
  endif()

  set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g -DNDEBUG")
  set(CMAKE_C_FLAGS_RELWITHDEBINFO "-O3 -g -DNDEBUG")
endif()
