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

#include <imgdnn.h>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "lite/core/op_lite.h"
#include "lite/utils/macros.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace imagination_nna {

struct TensorInfo {
  imgdnn_type type;
  std::vector<float> scales;
  std::vector<int> zero_points;
  DataLayoutType layout;
  unsigned count;
  unsigned axis;
};

bool HasInputArg(const OpInfo* op_info,
                 const Scope* scope,
                 const std::string& argname);
bool isScalesPerChannel(std::vector<float> scales);

void TensorInfoReset(TensorInfo* qnt);

}  // namespace imagination_nna
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle
