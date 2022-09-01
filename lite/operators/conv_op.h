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
#ifdef LITE_WITH_PROFILE
#include "lite/api/paddle_place.h"
#endif

namespace paddle {
namespace lite {
namespace operators {

class ConvOpLite : public OpLite {
 public:
  ConvOpLite() {}

  explicit ConvOpLite(const std::string& type) : OpLite(type) {}

  bool CheckShape() const override;
  bool InferShapeImpl() const override;
  bool InferShapeWithCache() const override { return true; }

#ifdef LITE_WITH_PROFILE
  void GetOpRuntimeInfo(paddle::lite::profile::OpCharacter* ch) {
    auto filter_dims = param_.filter->dims();
    auto input_dims = param_.x->dims();
    auto output_dims = param_.output->dims();
    ch->input_shape = ch->DimToStr(input_dims);
    ch->output_shape = ch->DimToStr(output_dims);
    ch->filter_shape = ch->DimToStr(filter_dims);
    ch->remark =
        std::to_string(filter_dims[2]) + "x" + std::to_string(filter_dims[3]) +
        "p" + std::to_string((*param_.paddings)[0]) + "s" +
        std::to_string(param_.strides[0]) + "g" +
        std::to_string(param_.groups) + "d" +
        std::to_string((*param_.dilations)[0]) + (param_.bias ? "Bias" : "") +
        ActivationTypeToStr(param_.activation_param.active_type);
    // MACs = 2.f * kw * kh * batchsize * out_c * out_h * out_w * in_c / group
    // GMACs = 1e-9f * MACs
    // GMACPS = 1e-6f * MACs / predict_ms
    ch->macs = 2.f * filter_dims[2] * filter_dims[3] *
               output_dims.production() * input_dims[1] / param_.groups;

    if (!param_.fuse_elementwise_op_type.empty()) {
      ch->remark += param_.fuse_elementwise_op_type;
      ch->macs += 1.0f * output_dims.production();
    }
  }
#endif

  bool AttachInput(const cpp::OpDescWrite& op_desc,
                   lite::Scope* scope) override {
    auto X = op_desc.Input("Input").front();
    param_.x = scope->FindVar(X)->GetMutable<lite::Tensor>();
    CHECK(param_.x);
    return true;
  }

  // TODO(Superjomn) replace framework::OpDesc with a lite one.
  bool AttachImpl(const cpp::OpDesc& op_desc, lite::Scope* scope) override {
    auto X = op_desc.Input("Input").front();
    auto Filter = op_desc.Input("Filter").front();
    auto Out = op_desc.Output("Output").front();

    param_.x = scope->FindVar(X)->GetMutable<lite::Tensor>();
    param_.filter = scope->FindVar(Filter)->GetMutable<lite::Tensor>();
    param_.output = scope->FindVar(Out)->GetMutable<lite::Tensor>();
    CHECK(param_.x);
    CHECK(param_.filter);
    CHECK(param_.output);
    input_tensor_ptrs_cache_.push_back(param_.x);
    output_tensor_ptrs_cache_.push_back(param_.output);

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
    if (std::find(input_arg_names.begin(),
                  input_arg_names.end(),
                  "ResidualData") != input_arg_names.end()) {
      auto res_data_arguments = op_desc.Input("ResidualData");
      if (res_data_arguments.size() > 0) {
        auto residual_data_var = scope->FindVar(res_data_arguments.front());
        if (residual_data_var != nullptr) {
          param_.residualData = const_cast<lite::Tensor*>(
              &(residual_data_var->Get<lite::Tensor>()));
        }
      }
    }

    if (op_desc.HasAttr("with_act") && op_desc.GetAttr<bool>("with_act")) {
      param_.activation_param.has_active = true;
      auto act_type = op_desc.GetAttr<std::string>("act_type");
      if (act_type == "relu") {
        param_.activation_param.active_type = lite_api::ActivationType::kRelu;
        param_.fuse_relu = true;
      } else if (act_type == "sigmoid") {
        param_.activation_param.active_type =
            lite_api::ActivationType::kSigmoid;
        param_.fuse_sigmoid = true;
      } else if (act_type == "tanh") {
        param_.activation_param.active_type = lite_api::ActivationType::kTanh;
        param_.fuse_tanh = true;
      } else if (act_type == "swish") {
        param_.activation_param.swish_scale =
            op_desc.GetAttr<float>("swish_scale");
        param_.activation_param.active_type = lite_api::ActivationType::kSwish;
        param_.fuse_swish = true;
      } else if (act_type == "abs") {
        param_.activation_param.active_type = lite_api::ActivationType::kAbs;
        param_.fuse_abs = true;
      } else if (act_type == "relu6") {
        param_.activation_param.active_type = lite_api::ActivationType::kRelu6;
        param_.activation_param.Relu_clipped_coef =
            op_desc.GetAttr<float>("fuse_brelu_threshold");  // 6.f
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
                      "relu, hard_swish, while the given activation type is "
                   << act_type;
      }
    }
    if (op_desc.HasAttr("scale_activation_type")) {
      param_.scale_activation_type =
          op_desc.GetAttr<std::string>("scale_activation_type");
    }

    if (op_desc.HasAttr("fuse_elementwise_op_type")) {
      param_.fuse_elementwise_op_type =
          op_desc.GetAttr<std::string>("fuse_elementwise_op_type");
      auto X = op_desc.Input("SecondInput").front();
      param_.second_x =
          const_cast<lite::Tensor*>(&(scope->FindVar(X)->Get<lite::Tensor>()));
    }

    if (op_desc.HasAttr("padding_algorithm")) {
      padding_algorithm_ = op_desc.GetAttr<std::string>("padding_algorithm");
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

#ifdef LITE_WITH_FPGA
    if (op_info != nullptr && op_info->HasAttr("fpga_static_quant")) {
      param_.enable_int8 = op_info->GetAttr<bool>("fpga_static_quant");
      auto input_scale_name = "Input0_scale";
      if (op_info->HasInputScale(input_scale_name, true)) {
        param_.input_scale = op_info->GetInputScale(input_scale_name, true)[0];
      }
    }
#endif

#ifdef LITE_WITH_FPGA
    if (std::find(input_arg_names.begin(), input_arg_names.end(), "Scale") !=
        input_arg_names.end()) {
      auto scale_arguments = op_desc.Input("Scale");
      if (scale_arguments.size() > 0) {
        auto scale_var = scope->FindVar(scale_arguments.front());
        if (scale_var != nullptr) {
          param_.scale =
              const_cast<lite::Tensor*>(&(scale_var->Get<lite::Tensor>()));
        }
      }
    }
#endif

    // conv3d: 3-pad to 6-pad, or conv2d: 2-pad to 4-pad
    if (paddings.size() == 2L || paddings.size() == 3L) {
      for (size_t i = 0; i < param_.strides.size(); ++i) {
        int copy_pad = *(paddings.begin() + 2 * i);
        paddings.insert(paddings.begin() + 2 * i + 1, copy_pad);
      }
    } else {
      if (paddings.size() != 4L && paddings.size() != 6L) {
        LOG(FATAL)
            << "Paddings size should be the same or twice as the input size.";
      }
    }
    param_.paddings = std::make_shared<std::vector<int>>(paddings);
    return true;
  }

  void AttachKernel(KernelBase* kernel) override { kernel->SetParam(param_); }

  std::string DebugString() const override { return "conv2d"; }

 protected:
  mutable ConvParam param_;
  std::string padding_algorithm_{""};
};
// update padding dilation
void UpdatePaddingAndDilation(std::vector<int>* paddings,
                              std::vector<int>* dilations,
                              const std::vector<int>& strides,
                              const std::string padding_algorithm,
                              const lite::DDim data_dims,
                              const lite::DDim& ksize);
}  // namespace operators
}  // namespace lite
}  // namespace paddle
