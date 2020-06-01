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

#include "lite/operators/reduce_mean_op.h"
#include <algorithm>
#include <string>
#include <vector>
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool ReduceMeanOp::CheckShape() const {
  CHECK_OR_FALSE(param_.X);
  CHECK_OR_FALSE(param_.Out);
  auto dims = param_.dim;
  auto x_dims = param_.X->dims();
  int x_rank = x_dims.size();
  if (dims.size() != 0) {
    for (size_t i = 0; i < dims.size(); i++) {
      if (dims[i] < 0) {
        dims[i] = x_rank + dims[i];
      }
      CHECK_OR_FALSE(dims[i] <= x_rank && dims[i] >= -x_rank);
    }
  }
  return true;
}

bool ReduceMeanOp::InferShapeImpl() const {
  auto dims = param_.dim;
  auto x_dims = param_.X->dims();
  bool reduce_all = false;
  bool keep_dim = param_.keep_dim;
  auto x_rank = x_dims.size();
  if (dims.size() != 0) {
    for (size_t i = 0; i < dims.size(); i++) {
      if (dims[i] < 0) {
        dims[i] = x_rank + dims[i];
      }
    }
  }
  std::stable_sort(dims.begin(), dims.end());
  if (dims.size() == 0) {
    reduce_all = true;
  }
  std::vector<int64_t> out_dims;
  if (reduce_all) {
    if (keep_dim) {
      out_dims.push_back(x_rank);
      out_dims.push_back(1);
    } else {
      out_dims.push_back(1);
    }
  } else {
    for (size_t i = 0; i < x_dims.size(); i++) {
      out_dims.push_back(x_dims[i]);
    }
    if (keep_dim) {
      for (size_t i = 0; i < dims.size(); ++i) {
        out_dims[dims[i]] = 1;
      }
    } else {
      const int64_t kDelFlag = -2;
      for (size_t i = 0; i < dims.size(); ++i) {
        out_dims[dims[i]] = kDelFlag;
      }
      out_dims.erase(remove(out_dims.begin(), out_dims.end(), kDelFlag),
                     out_dims.end());
    }
    param_.Out->Resize(DDim(out_dims));
    if (dims[0] != 0) {
      // Only pass LoD when not reducing on the first dim.
      *param_.Out->mutable_lod() = param_.X->lod();
    }
  }
  return true;
}

bool ReduceMeanOp::AttachImpl(const cpp::OpDesc &opdesc, lite::Scope *scope) {
  param_.X = const_cast<lite::Tensor *>(
      &scope->FindVar(opdesc.Input("X").front())->Get<lite::Tensor>());
  param_.Out =
      scope->FindVar(opdesc.Output("Out").front())->GetMutable<lite::Tensor>();
  param_.dim = opdesc.GetAttr<std::vector<int>>("dim");
  if (opdesc.HasAttr("keep_dim")) {
    param_.keep_dim = opdesc.GetAttr<bool>("keep_dim");
  } else {
    param_.keep_dim = false;
  }
  CHECK(param_.X);
  CHECK(param_.Out);
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(reduce_mean, paddle::lite::operators::ReduceMeanOp);
