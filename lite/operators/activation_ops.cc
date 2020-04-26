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
    // relu
    param_.active_type = lite_api::ActivationType::kRelu;
  } else if (opdesc.Type() == "leaky_relu") {
    // leaky_relu
    param_.Leaky_relu_alpha = opdesc.GetAttr<float>("alpha");
    param_.active_type = lite_api::ActivationType::kLeakyRelu;
  } else if (opdesc.Type() == "relu_clipped") {
    // relu_clipped
    param_.Relu_clipped_coef = opdesc.GetAttr<float>("Relu_clipped_coef");
  } else if (opdesc.Type() == "prelu") {
    // prelu
    param_.Prelu_mode = opdesc.GetAttr<std::string>("mode");
    auto prelu_alpha_name = opdesc.Input("Alpha").front();
    param_.Prelu_alpha =
        scope->FindVar(prelu_alpha_name)->GetMutable<lite::Tensor>();
    param_.active_type = lite_api::ActivationType::kPRelu;
  } else if (opdesc.Type() == "swish") {
    // swish
    param_.Swish_beta = opdesc.GetAttr<float>("beta");
    param_.active_type = lite_api::ActivationType::kSwish;
  } else if (opdesc.Type() == "hard_sigmoid") {
    // hard_sigomid
    param_.hard_sigmoid_slope = opdesc.GetAttr<float>("slope");
    param_.hard_sigmoid_offset = opdesc.GetAttr<float>("offset");
  } else if (opdesc.Type() == "sigmoid") {
    // sigmoid
    param_.active_type = lite_api::ActivationType::kSigmoid;
  } else if (opdesc.Type() == "tanh") {
    // tanh
    param_.active_type = lite_api::ActivationType::kTanh;
  } else if (opdesc.Type() == "exp") {
    // exp
    param_.active_type = lite_api::ActivationType::kExp;
  } else if (opdesc.Type() == "abs") {
    // abs
    param_.active_type = lite_api::ActivationType::kAbs;
  } else if (opdesc.Type() == "hard_swish") {
    // hard_swish
    param_.active_type = lite_api::ActivationType::kHardSwish;
    param_.hard_swish_threshold = opdesc.GetAttr<float>("threshold");
    param_.hard_swish_scale = opdesc.GetAttr<float>("scale");
    param_.hard_swish_offset = opdesc.GetAttr<float>("offset");
  } else if (opdesc.Type() == "reciprocal") {
    param_.active_type = lite_api::ActivationType::kReciprocal;
  }
  VLOG(4) << "opdesc.Type():" << opdesc.Type();

  param_.Out = scope->FindVar(out_name)->GetMutable<lite::Tensor>();
  return true;
}

#ifdef LITE_WITH_PROFILE
float ActivationOp::GetGops(){
  // todo
  auto act_type = param_.active_type;
  float gops = static_cast<float>(param_.X->numel());
  switch(act_type){
    case lite_api::ActivationType::kRelu:
      return gops;
    case lite_api::ActivationType::kRelu6:
      return 2.0 * gops;
    case lite_api::ActivationType::kLeakyRelu:
      return 2.0 * gops;
    case lite_api::ActivationType::kPRelu:
      return 2.0 * gops;
    case lite_api::ActivationType::kSwish:
      return 4.0 * gops;
    case lite_api::ActivationType::kSigmoid:
      return 3.0 * gops;
    case lite_api::ActivationType::kTanh:
      return 5.0 * gops;
    case lite_api::ActivationType::kExp:
      return gops;
    case lite_api::ActivationType::kAbs:
      return gops;
    case lite_api::ActivationType::kHardSwish:
      return 5.0 * gops;
    case lite_api::ActivationType::kReciprocal:
      return gops;
    default:
      std::cout << "This Type :" << static_cast<int>(act_type) << "doesn't support! " << std::endl;
      return gops;
  }
}
#endif

}  // namespace operators
}  // namespace lite
}  // namespace paddle

// Baisc activation ops
REGISTER_LITE_OP(sigmoid, paddle::lite::operators::ActivationOp);
REGISTER_LITE_OP(tanh, paddle::lite::operators::ActivationOp);
REGISTER_LITE_OP(relu, paddle::lite::operators::ActivationOp);
REGISTER_LITE_OP(leaky_relu, paddle::lite::operators::ActivationOp);
REGISTER_LITE_OP(relu6, paddle::lite::operators::ActivationOp);
REGISTER_LITE_OP(prelu, paddle::lite::operators::ActivationOp);
