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

#include "lite/operators/reduce_ops.h"
#include <algorithm>
#include <set>
#include "lite/core/op_registry.h"
namespace paddle {
namespace lite {
namespace operators {

bool ReduceOp::CheckShape() const {
  CHECK_OR_FALSE(param_.X);
  CHECK_OR_FALSE(param_.Out);
  auto dims = param_.dim;
  auto x_dims = param_.X->dims();
  int x_rank = x_dims.size();
  if (dims.size() != 0) {
    for (int i = 0; i < dims.size(); i++) {
      if (dims[i] < 0) {
        dims[i] = x_rank + dims[i];
      }
      CHECK_OR_FALSE(dims[i] <= x_rank && dims[i] >= -x_rank);
    }
  }
  return true;
}

bool ReduceOp::InferShapeImpl() const {
  const auto &x_dims = param_.X->dims();
  size_t x_rank = x_dims.size();
  auto dims = param_.dim;
  bool reduce_all = param_.reduce_all;
  bool keep_dim = param_.keep_dim;

  for (size_t i = 0; i < dims.size(); ++i) {
    if (dims[i] < 0) {
      dims[i] = x_rank + dims[i];
    }
    CHECK_LT(dims[i], x_rank)
        << "The dim should be in the range [-rank(input), rank(input).";
  }

  std::set<int> dims_set(dims.begin(), dims.end());
  bool full_dim = true;
  for (size_t i = 0; i < x_rank; i++) {
    if (dims_set.find(i) == dims_set.end()) {
      full_dim = false;
      break;
    }
  }
  reduce_all = (reduce_all || full_dim);

  if (reduce_all) {
    if (keep_dim)
      param_.Out->Resize(std::vector<int64_t>(x_rank, 1));
    else
      param_.Out->Resize(std::vector<int64_t>{1});
  } else {
    size_t out_rank = keep_dim ? x_rank : x_rank - dims.size();
    std::vector<DDim::value_type> out_dims(out_rank);
    std::stable_sort(dims.begin(), dims.end());
    int dim_index = 0;
    int out_index = 0;
    for (size_t i = 0; i < x_rank; ++i) {
      if (dim_index < dims.size() &&
          dims[dim_index] == static_cast<DDim::value_type>(i)) {
        if (keep_dim) {
          out_dims[out_index++] = 1;
        }
        dim_index++;
      } else {
        out_dims[out_index++] = x_dims[i];
      }
    }
    param_.Out->Resize(out_dims);
    if (dims[0] != 0) {
      param_.Out->set_lod(param_.X->lod());
    }
  }
  return true;
}

bool ReduceOp::AttachImpl(const cpp::OpDesc &opdesc, lite::Scope *scope) {
  param_.X = scope->FindTensor(opdesc.Input("X").front());
  param_.Out = scope->FindMutableTensor(opdesc.Output("Out").front());

  param_.dim = opdesc.GetAttr<std::vector<int>>("dim");
  if (opdesc.HasAttr("reduce_all")) {
    param_.reduce_all = opdesc.GetAttr<bool>("reduce_all");
  }
  if (opdesc.HasAttr("keep_dim")) {
    param_.keep_dim = opdesc.GetAttr<bool>("keep_dim");
  }
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

#ifdef LITE_BUILD_EXTRA
REGISTER_LITE_OP(reduce_sum, paddle::lite::operators::ReduceOp);
REGISTER_LITE_OP(reduce_prod, paddle::lite::operators::ReduceOp);
REGISTER_LITE_OP(reduce_max, paddle::lite::operators::ReduceOp);
REGISTER_LITE_OP(reduce_min, paddle::lite::operators::ReduceOp);
REGISTER_LITE_OP(reduce_all, paddle::lite::operators::ReduceOp);
REGISTER_LITE_OP(reduce_any, paddle::lite::operators::ReduceOp);
#endif  // LITE_BUILD_EXTRA

REGISTER_LITE_OP(reduce_mean, paddle::lite::operators::ReduceOp);
