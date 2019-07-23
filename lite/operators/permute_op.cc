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

#include "lite/operators/permute_op.h"
#include "lite/core/op_lite.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool PermuteOpLite::CheckShape() const {
  CHECK_OR_FALSE(param_.X.size());
  CHECK_OR_FALSE(param_.Out.size());
  return true;
}

bool PermuteOpLite::InferShape() const {
  std::vector<lite::DDim> input_dims;
  for (auto p : param_.X) {
    input_dims.push_back(p->dims());
  }
  const size_t n = input_dims.size();

  for (int i = 0; i < n; i++) {
    auto &out_dims = input_dims[i];
    CHECK_EQ(input_dims[i].size(), param_.order.size())
        << "permute order param is not valid";
    for (int j = 0; j < param_.order.size(); j++) {
      out_dims[j] = input_dims[i][param_.order[j]];
    }
    param_.Out[i]->Resize(lite::DDim(out_dims));
  }
  return true;
}

// TODO(Superjomn) replace framework::OpDesc with a lite one.
bool PermuteOpLite::AttachImpl(const cpp::OpDesc &op_desc, lite::Scope *scope) {
  auto inputs = op_desc.Input("X");
  auto out = op_desc.Output("Out");

  for (auto var : inputs) {
    param_.X.push_back(scope->FindVar(var)->GetMutable<lite::Tensor>());
  }
  for (auto var : out) {
    param_.Out.push_back(scope->FindVar(var)->GetMutable<lite::Tensor>());
  }
  param_.order = op_desc.GetAttr<std::vector<int>>("Order");

  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(permute, paddle::lite::operators::PermuteOpLite);
