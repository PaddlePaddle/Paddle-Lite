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

// paddle op reduce_all default is true, so we need recompute
bool recompute_reduce_all(const Tensor* x,
                          const std::vector<int>& dims,
                          bool reduce_all = false) {
  if (dims.size() == 0 || x->dims().size() == 0 ||
      dims.size() == x->dims().size() || reduce_all) {
    // when input 0D, it can only reduce_all
    return true;
  } else {
    return false;
  }
}

bool ReduceOp::CheckShape() const {
  CHECK_OR_FALSE(param_.X);
  CHECK_OR_FALSE(param_.Out);
  auto dims = param_.dim;
  auto x_dims = param_.X->dims();
  int x_rank = x_dims.size();
  return x_rank >= 0;
}

bool ReduceOp::InferShapeImpl() const {
  const auto& x_dims = param_.X->dims();
  int x_rank = x_dims.size();
  auto dims = param_.dim;
  bool& reduce_all = param_.reduce_all;
  bool keep_dim = param_.keep_dim;

  for (int i = 0; i < dims.size(); i++) {
    if (dims[i] < 0) {
      dims[i] = x_rank + dims[i] >= 0 ? x_rank + dims[i] : 0;
    }
  }
  // recompute reduce_all
  reduce_all = recompute_reduce_all(param_.X, dims, reduce_all);
  std::set<int> dims_set(dims.begin(), dims.end());
  bool full_dim = true;
  for (size_t i = 0; i < x_rank; i++) {
    if (dims_set.find(i) == dims_set.end() &&
        dims_set.find(i - x_rank) == dims_set.end()) {
      full_dim = false;
      break;
    }
  }
  reduce_all = (reduce_all || full_dim);
  if (reduce_all) {
    if (keep_dim)
      param_.Out->Resize(std::vector<int64_t>(x_rank, 1));
    else {
      param_.Out->Resize(std::vector<int64_t>({}));
    }
  } else {
    std::vector<int64_t> dims_vector(x_rank, 1);
    for (int i = 0; i < x_rank; i++) dims_vector[i] = x_dims[i];

    if (keep_dim) {
      for (size_t i = 0; i < dims.size(); ++i) {
        dims_vector[dims[i]] = 1;
      }
    } else {
      const int kDelFlag = -2;
      for (size_t i = 0; i < dims.size(); ++i) {
        dims_vector[dims[i]] = kDelFlag;
      }
      dims_vector.erase(
          remove(dims_vector.begin(), dims_vector.end(), kDelFlag),
          dims_vector.end());
    }
    if (!keep_dim && dims_vector.size() == 0) {
      dims_vector.push_back(1);
    }
    param_.Out->Resize(dims_vector);
    if (dims.size() > 0 && dims[0] != 0) {
      // Only pass LoD when not reducing on the first dim.
      param_.Out->set_lod(param_.X->lod());
    }
  }
  return true;
}

bool ReduceOp::AttachImpl(const cpp::OpDesc& opdesc, lite::Scope* scope) {
  param_.X = scope->FindTensor(opdesc.Input("X").front());
  param_.Out = scope->FindMutableTensor(opdesc.Output("Out").front());

  param_.dim = opdesc.GetAttr<std::vector<int>>("dim");
  if (opdesc.HasAttr("reduce_all")) {
    param_.reduce_all = opdesc.GetAttr<bool>("reduce_all");
  }
  if (opdesc.HasAttr("keep_dim")) {
    param_.keep_dim = opdesc.GetAttr<bool>("keep_dim");
  }

#ifdef LITE_WITH_XPU
  if (opdesc.HasAttr("enable_int8") && opdesc.GetAttr<bool>("enable_int8")) {
    param_.enable_int8 = true;
    param_.input_scale = opdesc.GetAttr<std::vector<float>>("X0_scale")[0];
    param_.output_scale = opdesc.GetAttr<std::vector<float>>("Out0_scale")[0];
  }
#endif

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
