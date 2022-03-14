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
// limitations under the License.i

#include "lite/operators/activation_ops.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool ActivationOp::CheckShape() const {
  CHECK_OR_FALSE(param_.X);
  CHECK_OR_FALSE(param_.Out);
  return true;
}

bool ActivationOp::InferShapeImpl() const {
  param_.Out->Resize(param_.X->dims());
  auto out_lod = param_.Out->mutable_lod();
  *out_lod = param_.X->lod();
  return true;
}

bool ActivationOp::AttachImpl(const cpp::OpDesc& opdesc, lite::Scope* scope) {
  auto x_name = opdesc.Input("X").front();
  auto out_name = opdesc.Output("Out").front();
  param_.X = scope->FindVar(x_name)->GetMutable<lite::Tensor>();

  if (opdesc.Type() == "relu") {
    param_.active_type = lite_api::ActivationType::kRelu;
  } else if (opdesc.Type() == "leaky_relu") {
    param_.Leaky_relu_alpha = opdesc.GetAttr<float>("alpha");
    param_.active_type = lite_api::ActivationType::kLeakyRelu;
  } else if (opdesc.Type() == "relu_clipped") {
    param_.Relu_clipped_coef = opdesc.GetAttr<float>("Relu_clipped_coef");
  } else if (opdesc.Type() == "prelu") {
    param_.Prelu_mode = opdesc.GetAttr<std::string>("mode");
    auto prelu_alpha_name = opdesc.Input("Alpha").front();
    param_.Prelu_alpha =
        scope->FindVar(prelu_alpha_name)->GetMutable<lite::Tensor>();
    param_.active_type = lite_api::ActivationType::kPRelu;
  } else if (opdesc.Type() == "swish") {
    param_.Swish_beta = opdesc.GetAttr<float>("beta");
    param_.active_type = lite_api::ActivationType::kSwish;
  } else if (opdesc.Type() == "hard_sigmoid") {
    param_.active_type = lite_api::ActivationType::kHardSigmoid;
    param_.hard_sigmoid_slope = opdesc.GetAttr<float>("slope");
    param_.hard_sigmoid_offset = opdesc.GetAttr<float>("offset");
  } else if (opdesc.Type() == "sigmoid") {
    param_.active_type = lite_api::ActivationType::kSigmoid;
  } else if (opdesc.Type() == "tanh") {
    param_.active_type = lite_api::ActivationType::kTanh;
  } else if (opdesc.Type() == "exp") {
    param_.active_type = lite_api::ActivationType::kExp;
  } else if (opdesc.Type() == "log") {
    param_.active_type = lite_api::ActivationType::kLog;
  } else if (opdesc.Type() == "abs") {
    param_.active_type = lite_api::ActivationType::kAbs;
  } else if (opdesc.Type() == "hard_swish") {
    param_.active_type = lite_api::ActivationType::kHardSwish;
    param_.hard_swish_threshold = opdesc.GetAttr<float>("threshold");
    param_.hard_swish_scale = opdesc.GetAttr<float>("scale");
    param_.hard_swish_offset = opdesc.GetAttr<float>("offset");
  } else if (opdesc.Type() == "reciprocal") {
    param_.active_type = lite_api::ActivationType::kReciprocal;
  } else if (opdesc.Type() == "thresholded_relu") {
    param_.active_type = lite_api::ActivationType::kThresholdedRelu;
    param_.relu_threshold = opdesc.GetAttr<float>("threshold");
  } else if (opdesc.Type() == "elu") {
    param_.active_type = lite_api::ActivationType::kElu;
    param_.Elu_alpha = opdesc.GetAttr<float>("alpha");
  } else if (opdesc.Type() == "relu6") {
    param_.active_type = lite_api::ActivationType::kRelu6;
    param_.threshold = opdesc.GetAttr<float>("threshold");
  } else if (opdesc.Type() == "gelu") {
    param_.active_type = lite_api::ActivationType::kGelu;
    if (opdesc.HasAttr("approximate")) {
      param_.gelu_approximate = opdesc.GetAttr<bool>("approximate");
    }
  } else if (opdesc.Type() == "erf") {
    param_.active_type = lite_api::ActivationType::kErf;
  } else if (opdesc.Type() == "sign") {
    param_.active_type = lite_api::ActivationType::kSign;
  } else if (opdesc.Type() == "softplus") {
    param_.active_type = lite_api::ActivationType::kSoftPlus;
    param_.softplus_threshold = opdesc.GetAttr<float>("threshold");
    param_.softplus_beta = opdesc.GetAttr<float>("beta");
  } else if (opdesc.Type() == "mish") {
    param_.active_type = lite_api::ActivationType::kMish;
    param_.threshold = opdesc.GetAttr<float>("threshold");
  }

  VLOG(4) << "opdesc.Type():" << opdesc.Type();

  param_.Out = scope->FindVar(out_name)->GetMutable<lite::Tensor>();
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

#ifdef LITE_BUILD_EXTRA
REGISTER_LITE_OP(square, paddle::lite::operators::ActivationOp);
REGISTER_LITE_OP(relu_clipped, paddle::lite::operators::ActivationOp);
REGISTER_LITE_OP(swish, paddle::lite::operators::ActivationOp);
REGISTER_LITE_OP(log, paddle::lite::operators::ActivationOp);
REGISTER_LITE_OP(exp, paddle::lite::operators::ActivationOp);
REGISTER_LITE_OP(abs, paddle::lite::operators::ActivationOp);
REGISTER_LITE_OP(floor, paddle::lite::operators::ActivationOp);
REGISTER_LITE_OP(hard_sigmoid, paddle::lite::operators::ActivationOp);
REGISTER_LITE_OP(sqrt, paddle::lite::operators::ActivationOp);
REGISTER_LITE_OP(rsqrt, paddle::lite::operators::ActivationOp);
REGISTER_LITE_OP(softsign, paddle::lite::operators::ActivationOp);
REGISTER_LITE_OP(gelu, paddle::lite::operators::ActivationOp);
REGISTER_LITE_OP(hard_swish, paddle::lite::operators::ActivationOp);
REGISTER_LITE_OP(reciprocal, paddle::lite::operators::ActivationOp);
REGISTER_LITE_OP(mish, paddle::lite::operators::ActivationOp);
#endif  // LITE_BUILD_EXTRA

// Baisc activation ops
REGISTER_LITE_OP(sigmoid, paddle::lite::operators::ActivationOp);
REGISTER_LITE_OP(tanh, paddle::lite::operators::ActivationOp);
REGISTER_LITE_OP(relu, paddle::lite::operators::ActivationOp);
REGISTER_LITE_OP(leaky_relu, paddle::lite::operators::ActivationOp);
REGISTER_LITE_OP(relu6, paddle::lite::operators::ActivationOp);
REGISTER_LITE_OP(prelu, paddle::lite::operators::ActivationOp);
REGISTER_LITE_OP(thresholded_relu, paddle::lite::operators::ActivationOp);
REGISTER_LITE_OP(elu, paddle::lite::operators::ActivationOp);
REGISTER_LITE_OP(erf, paddle::lite::operators::ActivationOp);
REGISTER_LITE_OP(softplus, paddle::lite::operators::ActivationOp);
