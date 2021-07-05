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

#include "lite/operators/group_norm_op.h"
#include <string>
#include <vector>
#include "lite/core/op_lite.h"
#include "lite/core/op_registry.h"
#include "lite/core/tensor.h"

namespace paddle {
namespace lite_metal {
namespace operators {

bool GroupNormOp::CheckShape() const {
  CHECK_OR_FALSE(param_.x);
  CHECK_OR_FALSE(param_.scale);
  CHECK_OR_FALSE(param_.bias);
  CHECK_OR_FALSE(param_.out);
  CHECK_OR_FALSE(param_.saved_mean);
  CHECK_OR_FALSE(param_.saved_variance);
  auto x_dims = param_.x->dims();
  auto scale_dims = param_.scale->dims();
  auto bias_dims = param_.bias->dims();
  if (param_.channels == -1) {
    param_.channels = (param_.data_layout_str == "NCHW")
                          ? x_dims[1]
                          : x_dims[x_dims.size() - 1];
  }
  // only support NCHW
  CHECK_EQ(param_.data_layout_str, "NCHW") << "data_layout must be NCHW";
  CHECK(x_dims.size() >= 2 && x_dims.size() <= 5)
      << "Input X must have 2 to 5 dimensions.";
  CHECK_EQ(scale_dims.size(), 1UL) << "Input Scale must have 1 dimensions.";
  CHECK_EQ(bias_dims.size(), 1UL) << "Input Bias must have 1 dimensions.";
  CHECK_GT(param_.epsilon, 0.f) << "epsilon should be greater than 0.f";
  CHECK_LT(param_.epsilon, 0.01f) << "epsilon should be less than 0.01f";
  CHECK_LE(param_.groups, param_.channels)
      << "groups should be less than channels";
  CHECK_GE(param_.groups, 1) << "groups should be greater than 1";
  CHECK_EQ(param_.channels, scale_dims[0])
      << "The Input(Scale)'s first dimension size of Op(group_norm) must be "
         "equal to the number of channels";
  CHECK_EQ(param_.channels, bias_dims[0])
      << "The Input(Bias)'s first dimension size of Op(group_norm) must be "
         "equal to the number of channels";
  return true;
}

bool GroupNormOp::InferShapeImpl() const {
  auto x_dims = param_.x->dims();
  int64_t batch_size = x_dims[0];
  param_.saved_mean->Resize({batch_size, param_.groups});
  param_.saved_variance->Resize({batch_size, param_.groups});
  param_.out->Resize(x_dims);
  return true;
}

bool GroupNormOp::AttachImpl(const cpp::OpDesc& op_desc, lite_metal::Scope* scope) {
  param_.x = scope->FindVar(op_desc.Input("X").front())->GetMutable<Tensor>();
  param_.scale =
      scope->FindVar(op_desc.Input("Scale").front())->GetMutable<Tensor>();
  param_.bias =
      scope->FindVar(op_desc.Input("Bias").front())->GetMutable<Tensor>();
  if (op_desc.HasOutput("SavedMean")) {
    param_.saved_mean = scope->FindVar(op_desc.Output("SavedMean").front())
                            ->GetMutable<Tensor>();
  } else if (op_desc.HasOutput("Mean")) {
    param_.saved_mean =
        scope->FindVar(op_desc.Output("Mean").front())->GetMutable<Tensor>();
  }
  if (op_desc.HasOutput("SavedVariance")) {
    param_.saved_variance =
        scope->FindVar(op_desc.Output("SavedVariance").front())
            ->GetMutable<Tensor>();
  } else if (op_desc.HasOutput("Variance")) {
    param_.saved_variance = scope->FindVar(op_desc.Output("Variance").front())
                                ->GetMutable<Tensor>();
  }
  param_.out =
      scope->FindVar(op_desc.Output("Y").front())->GetMutable<Tensor>();
  if (op_desc.HasAttr("data_layout")) {
    param_.data_layout_str = op_desc.GetAttr<std::string>("data_layout");
  }
  param_.epsilon = op_desc.GetAttr<float>("epsilon");
  param_.groups = op_desc.GetAttr<int>("groups");
  if (op_desc.HasAttr("channels")) {
    param_.channels = op_desc.GetAttr<int>("channels");
  } else {
    param_.channels = -1;
  }

  return true;
}

} /* namespace operators */
} /* namespace lite_metal */
} /* namespace paddle */

REGISTER_LITE_OP(group_norm, paddle::lite_metal::operators::GroupNormOp);
