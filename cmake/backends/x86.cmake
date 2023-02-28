# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

if(LITE_WITH_X86)
  include(external/xbyak)     # download xbyak package
  include(external/xxhash)    # download install xxhash
  include(external/libxsmm)   # download, build, install libxsmm
  include(external/mklml)     # download mklml package
  include(external/mkldnn)    # download, build, install mkldnn
endif()
