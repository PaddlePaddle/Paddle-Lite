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
  CHECK_OR_FALSE(param_.out);
  CHECK_OR_FALSE(param_.saved_mean);
  CHECK_OR_FALSE(param_.saved_variance);

  auto x_dims = param_.x->dims();
  CHECK(x_dims.size() >= 2 && x_dims.size() <= 5)
      << "Input X must have 2 to 5 dimensions.";
  if (param_.scale != nullptr) {
    auto scale_dims = param_.scale->dims();
    CHECK_EQ(scale_dims.size(), 1UL) << "Input Scale must have 1 dimensions.";
    CHECK_EQ(scale_dims[0], x_dims[1]) << "ShapeError: the shape of scale must "
                                       << "equal to the channel of input.";
  }
  if (param_.bias != nullptr) {
    auto bias_dims = param_.bias->dims();
    CHECK_EQ(bias_dims.size(), 1UL) << "Input Bias must have 1 dimensions.";
    CHECK_EQ(bias_dims[0], x_dims[1]) << "ShapeError: the shape of bias must "
                                      << "equal to the channel of input.";
  }
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
  AttachInput(op_desc, scope, "X", false /*is_dispensable*/, &param_.x);
  AttachInput(op_desc, scope, "Scale", true, &param_.scale);
  AttachInput(op_desc, scope, "Bias", true, &param_.bias);
  AttachOutput(op_desc, scope, "SavedMean", false, &param_.saved_mean);
  AttachOutput(op_desc, scope, "SavedVariance", false, &param_.saved_variance);
  AttachOutput(op_desc, scope, "Y", false, &param_.out);
  param_.epsilon = op_desc.GetAttr<float>("epsilon");
  if (op_desc.HasAttr("activation_type")) {
    auto act_type = op_desc.GetAttr<std::string>("activation_type");
    param_.activation_type = act_type;
    if (act_type == "relu") {
      param_.fuse_relu = true;
    } else if (act_type == "relu6") {
      param_.alpha = op_desc.GetAttr<float>("alpha");  // 6.f
    } else if (act_type == "leaky_relu") {
      param_.alpha = op_desc.GetAttr<float>("alpha");
    } else {
      LOG(FATAL) << "unsupported Activation type: " << act_type
                 << " fuse not support";
    }
  }
  return true;
}

} /* namespace operators */
} /* namespace lite */
} /* namespace paddle */

REGISTER_LITE_OP(instance_norm, paddle::lite::operators::InstanceNormOp);
