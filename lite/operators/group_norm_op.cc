// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
namespace lite {
namespace operators {

bool GroupNormOp::CheckShape() const {
  CHECK_OR_FALSE(param_.x);
  CHECK_OR_FALSE(param_.out);
  CHECK_OR_FALSE(param_.saved_mean);
  CHECK_OR_FALSE(param_.saved_variance);
  auto x_dims = param_.x->dims();
  if (param_.channels == -1) {
    param_.channels = (param_.data_layout_str == "NCHW")
                          ? x_dims[1]
                          : x_dims[x_dims.size() - 1];
  }
  // only support NCHW
  CHECK_EQ(param_.data_layout_str, "NCHW") << "data_layout must be NCHW";
  CHECK(x_dims.size() >= 2 && x_dims.size() <= 5)
      << "Input X must have 2 to 5 dimensions.";

  if (param_.scale != nullptr) {
    auto scale_dims = param_.scale->dims();
    CHECK_EQ(scale_dims.size(), 1UL) << "Input Scale must have 1 dimensions.";
    CHECK_EQ(scale_dims[0], param_.channels)
        << "The Input(Scale)'s first dimension size of Op(group_norm) must be "
        << "equal to the number of channels";
  }
  if (param_.bias != nullptr) {
    auto bias_dims = param_.bias->dims();
    CHECK_EQ(bias_dims.size(), 1UL) << "Input Bias must have 1 dimensions.";
    CHECK_EQ(bias_dims[0], param_.channels)
        << "The Input(Bias)'s first dimension size of Op(group_norm) must be "
        << "equal to the number of channels";
  }

  CHECK_GT(param_.epsilon, 0.f) << "epsilon should be greater than 0.f";
  CHECK_GE(param_.groups, 1) << "groups should be greater than 1";
  CHECK_LE(param_.groups, param_.channels)
      << "groups should be less than channels";
  // The channels should be divisible by groups
  CHECK_EQ(param_.channels % param_.groups, 0)
      << "The channels should be divisible by groups";
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

bool GroupNormOp::AttachImpl(const cpp::OpDesc& op_desc, lite::Scope* scope) {
  AttachInput(op_desc, scope, "X", false /*is_dispensable*/, &param_.x);
  AttachInput(op_desc, scope, "Scale", true, &param_.scale);
  AttachInput(op_desc, scope, "Bias", true, &param_.bias);

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
} /* namespace lite */
} /* namespace paddle */

REGISTER_LITE_OP(group_norm, paddle::lite::operators::GroupNormOp);
