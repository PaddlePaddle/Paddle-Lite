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

class ConvElementwiseTreeOpLite : public OpLite {
 public:
  ConvElementwiseTreeOpLite() {}

  explicit ConvElementwiseTreeOpLite(const std::string& type) : OpLite(type) {}

  bool CheckShape() const override;
  bool InferShapeImpl() const override;

#ifdef LITE_WITH_PROFILE
  void GetOpRuntimeInfo(paddle::lite::profile::OpCharacter* ch) {
    auto filter_dims = param_.conv_param.filter->dims();
    auto input_dims = param_.x->dims();
    auto output_dims = param_.output->dims();
    ch->input_shape = ch->DimToStr(input_dims);
    ch->output_shape = ch->DimToStr(output_dims);
    ch->filter_shape = ch->DimToStr(filter_dims);
    ch->remark =
        std::to_string(filter_dims[2]) + "x" + std::to_string(filter_dims[3]) +
        "p" + std::to_string((*param_.conv_param.paddings)[0]) + "s" +
        std::to_string(param_.conv_param.strides[0]) + "g" +
        std::to_string(param_.conv_param.groups) + "d" +
        std::to_string((*param_.conv_param.dilations)[0]) +
        (param_.conv_param.bias ? "Bias" : "") +
        ActivationTypeToStr(param_.conv_param.activation_param.active_type);
    // MACs = 2.f * kw * kh * batchsize * out_c * out_h * out_w * in_c / group
    // GMACs = 1e-9f * MACs
    // GMACPS = 1e-6f * MACs / predict_ms
    ch->macs = 2.f * filter_dims[2] * filter_dims[3] *
               output_dims.production() * input_dims[1] /
               param_.conv_param.groups;

    if (!param_.conv_param.fuse_elementwise_op_type.empty()) {
      ch->remark += param_.conv_param.fuse_elementwise_op_type;
      ch->macs += 1.0f * output_dims.numel();
    }
  }
#endif

  // TODO(Superjomn) replace framework::OpDesc with a lite one.
  bool AttachImpl(const cpp::OpDesc& op_desc, lite::Scope* scope) override {
    AttachParam(&param_);
    auto X = op_desc.Input("Input0").front();
    auto Y = op_desc.Input("Input1").front();
    auto Filter = op_desc.Input("Filter").front();
    auto Out = op_desc.Output("Output").front();

    param_.x = scope->FindVar(X)->GetMutable<lite::Tensor>();
    param_.y = scope->FindVar(Y)->GetMutable<lite::Tensor>();
    param_.filter = scope->FindVar(Filter)->GetMutable<lite::Tensor>();
    param_.output = scope->FindVar(Out)->GetMutable<lite::Tensor>();

    param_.conv_param.x = param_.x;
    param_.conv_param.filter = param_.filter;
    param_.conv_param.strides = op_desc.GetAttr<std::vector<int>>("strides");
    std::vector<int> paddings = op_desc.GetAttr<std::vector<int>>("paddings");
    param_.conv_param.groups = op_desc.GetAttr<int>("groups");
    auto dilations = op_desc.GetAttr<std::vector<int>>("dilations");
    param_.conv_param.dilations = std::make_shared<std::vector<int>>(dilations);

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
          param_.conv_param.bias = param_.bias;
        }
      }
    }

    if (op_desc.HasAttr("has_conv_act") &&
        op_desc.GetAttr<bool>("has_conv_act")) {
      param_.has_conv_act = true;
      param_.conv_param.activation_param.has_active = true;
      auto act_type = op_desc.GetAttr<std::string>("conv_act_type");
      if (act_type == "relu") {
        param_.conv_param.activation_param.active_type =
            lite_api::ActivationType::kRelu;
        param_.conv_param.fuse_relu = true;
      } else if (act_type == "relu6") {
        param_.conv_param.activation_param.active_type =
            lite_api::ActivationType::kRelu6;
        param_.conv_param.activation_param.Relu_clipped_coef =
            op_desc.GetAttr<float>("conv_relu6");  // 6.f
      } else if (act_type == "leaky_relu") {
        param_.conv_param.activation_param.active_type =
            lite_api::ActivationType::kLeakyRelu;
        param_.conv_param.activation_param.Leaky_relu_alpha =
            op_desc.GetAttr<float>("conv_leaky_relu");
      } else {
        LOG(FATAL) << "The fused conv only supports fuse with relu, leaky "
                      "relu, hard_swish, while the given activation type is "
                   << act_type;
      }
    } else {
      param_.has_conv_act = false;
    }

    // 2-pad to 4-pad
    if (paddings.size() == 2L) {
      for (size_t i = 0; i < param_.conv_param.strides.size(); ++i) {
        int copy_pad = *(paddings.begin() + 2 * i);
        paddings.insert(paddings.begin() + 2 * i + 1, copy_pad);
      }
    } else {
      if (paddings.size() != 4L) {
        LOG(FATAL)
            << "Paddings size should be the same or twice as the input size.";
      }
    }
    param_.conv_param.paddings = std::make_shared<std::vector<int>>(paddings);

    // elementwise param
    param_.elt_param.Y = param_.y;
    param_.elt_param.Out = param_.output;
    param_.elt_param.axis = op_desc.GetAttr<int>("axis");
    if (op_desc.HasAttr("fuse_scale")) {
      param_.elt_param.fuse_scale = op_desc.GetAttr<bool>("fuse_scale");
      param_.elt_param.alpha = op_desc.GetAttr<float>("alpha");
      param_.elt_param.bias = op_desc.GetAttr<float>("bias");
    }
    if (op_desc.HasAttr("has_elt_act") &&
        op_desc.GetAttr<bool>("has_elt_act")) {
      param_.has_elt_act = true;
      param_.elt_param.act_type = op_desc.GetAttr<std::string>("elt_act_type");
    } else {
      param_.has_elt_act = false;
    }
    return true;
  }

  void AttachKernel(KernelBase* kernel) override { kernel->SetParam(param_); }

  std::string DebugString() const override { return "conv2d"; }

 private:
  mutable FusionConvElementParam param_;
  std::string padding_algorithm_{""};
};
}  // namespace operators
}  // namespace lite
}  // namespace paddle
