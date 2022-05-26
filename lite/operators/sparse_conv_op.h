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

#pragma once

#include <memory>
#include <string>
#include <vector>
#include "lite/core/kernel.h"
#include "lite/core/op_lite.h"
#include "lite/core/scope.h"
#include "lite/core/tensor.h"
#include "lite/operators/op_params.h"
#include "lite/utils/all.h"

namespace paddle {
namespace lite {
namespace operators {

class SparseConvOp : public OpLite {
 public:
  SparseConvOp() {}

  explicit SparseConvOp(const std::string& type) : OpLite(type) {}

  bool CheckShape() const override;

  bool InferShapeImpl() const override;

#ifdef LITE_WITH_PROFILE
  void GetOpRuntimeInfo(paddle::lite::profile::OpCharacter* ch) {
    auto filter_dims = param_.oc_nonzeros->dims();
    auto input_dims = param_.x->dims();
    auto output_dims = param_.output->dims();
    ch->input_shape = ch->DimToStr(input_dims);
    ch->output_shape = ch->DimToStr(output_dims);
    ch->filter_shape = ch->DimToStr(filter_dims) + "x" +
                       std::to_string(input_dims[1]) + "x1x1";
    ch->remark = std::to_string(1) + "x" + std::to_string(1) + "p" +
                 std::to_string((*param_.paddings)[0]) + "s" +
                 std::to_string(param_.strides[0]) + "g" +
                 std::to_string(param_.groups) + "d" +
                 std::to_string((*param_.dilations)[0]) +
                 (param_.bias ? "Bias" : "") +
                 ActivationTypeToStr(param_.activation_param.active_type);
    // MACs = 2.f * kw(1) * kh(1) * batchsize * out_c * out_h * out_w * in_c /
    // group
    // GMACs = 1e-9f * MACs
    // GMACPS = 1e-6f * MACs / predict_ms
    ch->macs = 2.f * output_dims.production() * input_dims[1] / param_.groups;
  }
#endif

  bool AttachImpl(const cpp::OpDesc& op_desc, lite::Scope* scope) override {
    auto X = op_desc.Input("Input").front();
    auto NonZeroWeights = op_desc.Input("NonZeroWeights").front();
    auto OcNonZeros = op_desc.Input("OcNonZeros").front();
    auto Diffs = op_desc.Input("Diffs").front();
    auto Out = op_desc.Output("Output").front();

    param_.x = scope->FindVar(X)->GetMutable<lite::Tensor>();
    param_.nonzero_weights =
        scope->FindVar(NonZeroWeights)->GetMutable<lite::Tensor>();
    param_.oc_nonzeros = scope->FindVar(OcNonZeros)->GetMutable<lite::Tensor>();
    param_.diffs = scope->FindVar(Diffs)->GetMutable<lite::Tensor>();
    param_.output = scope->FindVar(Out)->GetMutable<lite::Tensor>();

    param_.strides = op_desc.GetAttr<std::vector<int>>("strides");
    std::vector<int> paddings = op_desc.GetAttr<std::vector<int>>("paddings");
    param_.groups = op_desc.GetAttr<int>("groups");
    auto dilations = op_desc.GetAttr<std::vector<int>>("dilations");
    param_.dilations = std::make_shared<std::vector<int>>(dilations);

    // optional params
    std::vector<std::string> input_arg_names = op_desc.InputArgumentNames();
    if (std::find(input_arg_names.begin(), input_arg_names.end(), "Bias") !=
        input_arg_names.end()) {
      auto bias_arguments = op_desc.Input("Bias");
      if (bias_arguments.size() > 0) {
        auto bias_var = scope->FindVar(bias_arguments.front());
        if (bias_var != nullptr) {
          param_.bias =
              const_cast<lite::Tensor*>(&(bias_var->Get<lite::Tensor>()));
        }
      }
    }

    if (op_desc.HasAttr("with_act") && op_desc.GetAttr<bool>("with_act")) {
      param_.activation_param.has_active = true;
      auto act_type = op_desc.GetAttr<std::string>("act_type");
      if (act_type == "relu") {
        param_.activation_param.active_type = lite_api::ActivationType::kRelu;
        param_.fuse_relu = true;
      } else if (act_type == "relu6") {
        param_.activation_param.active_type = lite_api::ActivationType::kRelu6;
        param_.activation_param.Relu_clipped_coef =
            op_desc.GetAttr<float>("fuse_brelu_threshold");
      } else if (act_type == "leaky_relu") {
        param_.activation_param.active_type =
            lite_api::ActivationType::kLeakyRelu;
        param_.activation_param.Leaky_relu_alpha =
            op_desc.GetAttr<float>("leaky_relu_alpha");
      } else if (act_type == "hard_swish") {
        param_.activation_param.active_type =
            lite_api::ActivationType::kHardSwish;
        param_.activation_param.hard_swish_threshold =
            op_desc.GetAttr<float>("hard_swish_threshold");
        param_.activation_param.hard_swish_scale =
            op_desc.GetAttr<float>("hard_swish_scale");
        param_.activation_param.hard_swish_offset =
            op_desc.GetAttr<float>("hard_swish_offset");
      } else if (act_type == "hard_sigmoid") {
        param_.activation_param.active_type =
            lite_api::ActivationType::kHardSigmoid;
        param_.activation_param.hard_sigmoid_slope =
            op_desc.GetAttr<float>("slope");
        param_.activation_param.hard_sigmoid_offset =
            op_desc.GetAttr<float>("offset");
      } else if (act_type == "prelu") {
        param_.activation_param.active_type = lite_api::ActivationType::kPRelu;
        param_.activation_param.Prelu_mode =
            op_desc.GetAttr<std::string>("prelu_mode");
        auto prelu_alpha_name = op_desc.Input("Prelu_alpha").front();
        auto prelu_alpha_var = scope->FindVar(prelu_alpha_name);
        param_.activation_param.Prelu_alpha =
            const_cast<lite::Tensor*>(&(prelu_alpha_var->Get<lite::Tensor>()));
      } else {
        LOG(FATAL) << "The fused conv only supports fuse with relu, leaky "
                      "relu, relu6, while the given activation type is "
                   << act_type;
      }
    }
    if (op_desc.HasAttr("first_ic")) {
      param_.first_ic = op_desc.GetAttr<int>("first_ic");
    }
    if (op_desc.HasAttr("flag_semi")) {
      param_.flag_semi = op_desc.GetAttr<int>("flag_semi");
    }

    // For Int8
    const OpInfo* op_info = static_cast<const OpInfo*>(&op_desc);
    if (op_info != nullptr && op_info->HasAttr("enable_int8")) {
      param_.enable_int8 = op_info->GetAttr<bool>("enable_int8");
      auto input_scale_name = "Input0_scale";
      auto filter_scale_name = "Filter0_scale";
      auto output_scale_name = "Output0_scale";
      if (op_info->HasInputScale(input_scale_name, true))
        param_.input_scale = op_info->GetInputScale(input_scale_name, true)[0];
      if (op_info->HasInputScale(filter_scale_name, true))
        param_.weight_scale = op_info->GetInputScale(filter_scale_name, true);
      if (op_info->HasOutputScale(output_scale_name, true)) {
        param_.output_scale =
            op_info->GetOutputScale(output_scale_name, true)[0];
      }
    }
    // 2-pad to 4-pad
    if (paddings.size() == 2L) {
      for (size_t i = 0; i < param_.strides.size(); ++i) {
        int copy_pad = *(paddings.begin() + 2 * i);
        paddings.insert(paddings.begin() + 2 * i + 1, copy_pad);
      }
    } else {
      if (paddings.size() != 4L) {
        LOG(FATAL)
            << "Paddings size should be the same or twice as the input size.";
      }
    }
    param_.paddings = std::make_shared<std::vector<int>>(paddings);
    return true;
  }

  void AttachKernel(KernelBase* kernel) override { kernel->SetParam(param_); }

  std::string DebugString() const override { return "sparse_conv2d"; }

 private:
  mutable SparseConvParam param_;
};

}  // namespace operators
}  // namespace lite
}  // namespace paddle
