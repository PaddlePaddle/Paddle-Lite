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

#include "lite/operators/instance_norm_op.h"
#include <string>
#include <vector>
#include "lite/core/op_lite.h"
#include "lite/core/op_registry.h"
#include "lite/core/tensor.h"

namespace paddle {
namespace lite {
namespace operators {

bool InstanceNormOp::CheckShape() const {
  CHECK_OR_FALSE(param_.x);
  CHECK_OR_FALSE(param_.scale);
  CHECK_OR_FALSE(param_.bias);
  CHECK_OR_FALSE(param_.out);
  CHECK_OR_FALSE(param_.saved_mean);
  CHECK_OR_FALSE(param_.saved_variance);
  auto x_dims = param_.x->dims();
  auto scale_dims = param_.scale->dims();
  auto bias_dims = param_.bias->dims();
  CHECK(x_dims.size() >= 2 && x_dims.size() <= 5)
      << "Input X must have 2 to 5 dimensions.";
  CHECK_EQ(scale_dims.size(), 1UL) << "Input Scale must have 1 dimensions.";
  CHECK_EQ(bias_dims.size(), 1UL) << "Input Bias must have 1 dimensions.";
  CHECK_GT(param_.epsilon, 0.f) << "epsilon should be greater than 0.f";
  CHECK_LT(param_.epsilon, 0.01f) << "epsilon should be less than 0.01f";
  return true;
}

bool InstanceNormOp::InferShapeImpl() const {
  auto x_dims = param_.x->dims();
  int64_t batch_size = x_dims[0];
  int64_t channel_size = x_dims[1];
  param_.saved_mean->Resize({batch_size * channel_size});
  param_.saved_variance->Resize({batch_size * channel_size});
  param_.out->Resize(x_dims);
  return true;
}

bool InstanceNormOp::AttachImpl(const cpp::OpDesc& op_desc,
                                lite::Scope* scope) {
  param_.x = scope->FindVar(op_desc.Input("X").front())->GetMutable<Tensor>();
  param_.scale =
      scope->FindVar(op_desc.Input("Scale").front())->GetMutable<Tensor>();
  param_.bias =
      scope->FindVar(op_desc.Input("Bias").front())->GetMutable<Tensor>();
  param_.saved_mean =
      scope->FindVar(op_desc.Output("SavedMean").front())->GetMutable<Tensor>();
  param_.saved_variance =
      scope->FindVar(op_desc.Output("SavedVariance").front())
          ->GetMutable<Tensor>();
  param_.out =
      scope->FindVar(op_desc.Output("Y").front())->GetMutable<Tensor>();
  param_.epsilon = op_desc.GetAttr<float>("epsilon");
  return true;
}

} /* namespace operators */
} /* namespace lite */
} /* namespace paddle */

REGISTER_LITE_OP(instance_norm, paddle::lite::operators::InstanceNormOp);
