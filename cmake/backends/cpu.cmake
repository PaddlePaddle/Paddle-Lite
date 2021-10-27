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

include(external/mklml)     # download mklml package
include(external/xbyak)     # download xbyak package
include(external/libxsmm)   # download, build, install libxsmm
include(external/openblas)  # download, build, install openblas
include(external/mkldnn)    # download, build, install mkldnn
include(external/eigen)     # download eigen3
include(external/xxhash)    # download install xxhash needed for x86 jit

set(WITH_MKLML ${WITH_MKL})
if (NOT DEFINED WITH_MKLDNN)
    if (WITH_MKL AND AVX2_FOUND)
        set(WITH_MKLDNN ON)
    else()
        message(STATUS "Do not have AVX2 intrinsics and disabled MKL-DNN")
        set(WITH_MKLDNN OFF)
    endif()
endif()
