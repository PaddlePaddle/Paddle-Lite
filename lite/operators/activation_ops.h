// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once
#include <string>
#include "lite/core/op_lite.h"
#ifdef LITE_WITH_PROFILE
#include "lite/api/paddle_place.h"
#endif

namespace paddle {
namespace lite {
namespace operators {

class ActivationOp : public OpLite {
 public:
  explicit ActivationOp(const std::string& type) : OpLite(type) {}

  bool CheckShape() const override;

  bool InferShapeImpl() const override;

  bool AttachImpl(const cpp::OpDesc& opdesc, lite::Scope* scope) override;

  void AttachKernel(KernelBase* kernel) override { kernel->SetParam(param_); }

  std::string DebugString() const override { return "activation_op"; }

#ifdef LITE_WITH_PROFILE
  void GetOpRuntimeInfo(paddle::lite::profile::OpCharacter* ch) {
    auto input_dims = param_.X->dims();
    auto output_dims = param_.Out->dims();
    ch->input_shape = ch->DimToStr(input_dims);
    ch->output_shape = ch->DimToStr(output_dims);
    ch->remark = ActivationTypeToStr(param_.active_type);
    switch (param_.active_type) {
      case lite_api::ActivationType::kRelu:
        ch->macs = param_.X->numel();
        break;
      case lite_api::ActivationType::kRelu6:
        ch->macs = param_.X->numel() * 2.0;
        break;
      case lite_api::ActivationType::kLeakyRelu:
        ch->macs = param_.X->numel() * 2.0;
        break;
      case lite_api::ActivationType::kPRelu:
        ch->macs = param_.X->numel() * 2.0;
        break;
      case lite_api::ActivationType::kSwish:
        ch->macs = param_.X->numel() * 4.0;
        break;
      case lite_api::ActivationType::kSigmoid:
        ch->macs = param_.X->numel() * 3.0;
        break;
      case lite_api::ActivationType::kTanh:
        ch->macs = param_.X->numel() * 5.0;
        break;
      case lite_api::ActivationType::kExp:
        ch->macs = param_.X->numel();
        break;
      case lite_api::ActivationType::kAbs:
        ch->macs = param_.X->numel();
        break;
      case lite_api::ActivationType::kHardSwish:
        ch->macs = param_.X->numel() * 5.0;
        break;
      case lite_api::ActivationType::kReciprocal:
        ch->macs = param_.X->numel();
        break;
      case lite_api::ActivationType::kIndentity:
        break;
      case lite_api::ActivationType::kThresholdedRelu:
        ch->macs = param_.X->numel();
        break;
      case lite_api::ActivationType::kElu:
        ch->macs = param_.X->numel();
        break;
      default:
        LOG(FATAL) << "This Type of Activation:"
                   << static_cast<int>(param_.active_type)
                   << ActivationTypeToStr(param_.active_type)
                   << " doesn't support";
    }
  }
#endif

 private:
  mutable operators::ActivationParam param_;
};

}  // namespace operators
}  // namespace lite
}  // namespace paddle
