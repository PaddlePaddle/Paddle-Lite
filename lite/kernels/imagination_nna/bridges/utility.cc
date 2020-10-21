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

#include "lite/kernels/imagination_nna/bridges/utility.h"
#include <utility>

namespace paddle {
namespace lite {
namespace subgraph {
namespace imagination_nna {

bool HasInputArg(const OpInfo* op_info,
                 const Scope* scope,
                 const std::string& argname) {
  auto iarg_names = op_info->input_argnames();
  if (std::find(iarg_names.begin(), iarg_names.end(), argname) !=
      iarg_names.end()) {
    auto inputs = op_info->Input(argname);
    if (inputs.empty()) {
      return false;
    }
    auto var_name = inputs.front();
    auto var = scope->FindVar(var_name);
    return var != nullptr;
  } else {
    return false;
  }
}

bool isScalesPerChannel(std::vector<float> scales) {
  bool per_channel = false;
  for (std::vector<float>::iterator iter = scales.begin() + 1;
       iter != scales.end();
       iter++) {
    if (*iter != scales.at(0)) {
      per_channel = true;
      break;
    }
  }
  return per_channel;
}

void TensorInfoReset(TensorInfo* qnt) {
  qnt->count = 0;
  qnt->axis = 0;
  qnt->scales.clear();
  qnt->zero_points.clear();
  qnt->layout = DATALAYOUT(kNCHW);
}

}  // namespace imagination_nna
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle
