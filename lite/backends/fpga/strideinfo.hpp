// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <cstddef>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

namespace paddle {
namespace zynqmp {

struct StrideInfo {
  StrideInfo()
      : wd_enable_(false),
        wd_offset_(-1),
        fuse_idx_(-1),
        original_out_channel_(-1),
        start_idx_(0),
        end_idx_(0) {}
  bool wd_enable_;
  int wd_offset_;
  int fuse_idx_;
  int original_out_channel_;
  int start_idx_;
  int end_idx_;
};

}  // namespace lite
}  // namespace paddle
