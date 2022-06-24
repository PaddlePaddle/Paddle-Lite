# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
# http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# The configuration of this file applies only to NNAdapter and its driver HALs.

# Enable throwing exception when check failed in logging.cc
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Os -fvisibility=hidden -fvisibility-inlines-hidden -fexceptions -fasynchronous-unwind-tables -funwind-tables")

# Avoid the export symbols being stripped, which will cause 'undefined reference' errors when HAL is compiled independently.
string(REPLACE "-Wl,--exclude-libs,ALL" "" CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})

# CANN libraries only supports old ABI of libstdc++, so need to set -D_GLIBCXX_USE_CXX11_ABI=0
# and the common modules(utility and optimizer) needs to be compiled and linked separately if 
# the version of GCC greater than 5.0
if(NNADAPTER_WITH_HUAWEI_ASCEND_NPU)
  if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 5.0)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
  endif()
  # Resolve the compilation error caused by the introduction of ACL GE header file
  add_compile_options(-Wno-ignored-qualifiers)
endif()
