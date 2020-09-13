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
  }
#endif

  // TODO(Superjomn) replace framework::OpDesc with a lite one.
  bool AttachImpl(const cpp::OpDesc& op_desc, lite::Scope* scope) override {
    AttachParam(&param_);
    auto X = op_desc.Input("Input").front();
    auto Filter = op_desc.Input("Filter").front();
    auto Out = op_desc.Output("Output").front();

    param_.x = scope->FindVar(X)->GetMutable<lite::Tensor>();
    param_.filter = scope->FindVar(Filter)->GetMutable<lite::Tensor>();
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
      } else if (act_type == "relu6") {
        param_.activation_param.active_type = lite_api::ActivationType::kRelu6;
        param_.activation_param.Relu_clipped_coef =
            op_desc.GetAttr<float>("fuse_brelu_threshold");  // 6.f
      } else if (act_type == "leaky_relu") {
        param_.activation_param.active_type =
            lite_api::ActivationType::kLeakyRelu;
        param_.activation_param.Leaky_relu_alpha =
            op_desc.GetAttr<float>("leaky_relu_alpha");
      } else {
        CHECK(false)
            << "The fused conv only supports fuse with relu and leaky relu";
      }
    }

    if (op_desc.HasAttr("padding_algorithm")) {
      padding_algorithm_ = op_desc.GetAttr<std::string>("padding_algorithm");
    }
    // For Int8
    const OpInfo* op_info = dynamic_cast<const OpInfo*>(&op_desc);
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

  std::string DebugString() const override { return "conv2d"; }

 private:
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
