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

#include "lite/operators/fill_constant_batch_size_like_op.h"
#include <algorithm>
#include "lite/core/op_lite.h"
#include "lite/core/op_registry.h"
#include "lite/core/tensor.h"

namespace paddle {
namespace lite {
namespace operators {

bool FillConstantBatchSizeLikeOp::CheckShape() const {
  CHECK_OR_FALSE(param_.Input);
  CHECK_OR_FALSE(param_.Out);
  return true;
}

bool FillConstantBatchSizeLikeOp::InferShape() const {
  auto shape = param_.shape;
  std::vector<int64_t> shape_int64(shape.size(), 0);
  std::transform(shape.begin(), shape.end(), shape_int64.begin(), [](int a) {
    return static_cast<int64_t>(a);
  });
  lite::DDim output_dim(shape_int64);

  int input_dim_idx = param_.input_dim_idx;
  int output_dim_idx = param_.output_dim_idx;

  output_dim[output_dim_idx] = param_.Input->dims()[input_dim_idx];
  param_.Out->Resize(output_dim);
  return true;
}

bool FillConstantBatchSizeLikeOp::AttachImpl(const cpp::OpDesc &op_desc,
                                             lite::Scope *scope) {
  auto Input = op_desc.Input("X").front();
  auto Out = op_desc.Output("Out").front();
  param_.Input = scope->FindVar(Input)->GetMutable<lite::Tensor>();
  param_.Out = scope->FindVar(Out)->GetMutable<lite::Tensor>();
  param_.shape = op_desc.GetAttr<std::vector<int>>("shape");
  param_.input_dim_idx = op_desc.GetAttr<int>("input_dim_idx");
  param_.output_dim_idx = op_desc.GetAttr<int>("output_dim_idx");
  param_.dtype = op_desc.GetAttr<int>("dtype");
  param_.value = op_desc.GetAttr<float>("value");
  CHECK(param_.Input);
  CHECK(param_.Out);

  return true;
}

} /* namespace operators */
} /* namespace lite */
} /* namespace paddle */

REGISTER_LITE_OP(fill_constant_batch_size_like,
                 paddle::lite::operators::FillConstantBatchSizeLikeOp);
