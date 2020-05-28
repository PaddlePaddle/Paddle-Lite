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

#include "lite/operators/reduce_prod_op.h"
#include <algorithm>
#include <vector>
#include "lite/core/op_lite.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool ReduceProdOpLite::CheckShape() const {
  CHECK_OR_FALSE(param_.x);
  CHECK_OR_FALSE(param_.output);
  return true;
}

bool ReduceProdOpLite::InferShapeImpl() const {
  auto x = param_.x;
  auto out = param_.output;
  std::vector<int> dim = param_.dim;
  bool reduce_all = param_.reduce_all;
  bool keep_dim = param_.keep_dim;

  auto x_dims = x->dims();
  auto x_rank = x_dims.size();
  CHECK_OR_FALSE(x_rank <= 6U);
  for (size_t i = 0; i < dim.size(); i++) {
    if (dim[i] < 0) {
      dim[i] = x_rank + dim[i];
    }
    CHECK_OR_FALSE(static_cast<size_t>(dim[i]) < x_rank);
  }
  std::stable_sort(dim.begin(), dim.end());

  if (reduce_all || dim.size() == 0) {
    if (keep_dim) {
      out->Resize({static_cast<int64_t>(x_rank), 1});
    } else {
      out->Resize({1});
    }
  } else {
    auto dims_vector = x_dims.Vectorize();
    if (keep_dim) {
      for (size_t i = 0; i < dim.size(); ++i) {
        dims_vector[dim[i]] = 1;
      }
    } else {
      const int kDelFlag = -2;
      for (size_t i = 0; i < dim.size(); ++i) {
        dims_vector[dim[i]] = kDelFlag;
      }
      dims_vector.erase(
          std::remove(dims_vector.begin(), dims_vector.end(), kDelFlag),
          dims_vector.end());
    }
    if (!keep_dim && dims_vector.size() == 0) {
      dims_vector.push_back(1);
    }
    out->Resize(dims_vector);
    if (dim.size() > 0 && dim[0] != 0) {
      out->set_lod(x->lod());
    }
  }
  return true;
}

bool ReduceProdOpLite::AttachImpl(const cpp::OpDesc &op_desc,
                                  lite::Scope *scope) {
  auto x = op_desc.Input("X").front();
  param_.x = scope->FindVar(x)->GetMutable<lite::Tensor>();

  auto output = op_desc.Output("Out").front();
  param_.output = scope->FindVar(output)->GetMutable<lite::Tensor>();

  param_.dim = op_desc.GetAttr<std::vector<int>>("dim");
  param_.keep_dim = op_desc.GetAttr<bool>("keep_dim");
  param_.reduce_all = op_desc.GetAttr<bool>("reduce_all");
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(reduce_prod, paddle::lite::operators::ReduceProdOpLite);
