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
#include "lite/operators/conv_op.h"
#include "lite/operators/op_params.h"
#include "lite/utils/all.h"
#ifdef LITE_WITH_PROFILE
#include "lite/api/paddle_place.h"

#endif

namespace paddle {
namespace lite {
namespace operators {

class FpgaConvOpLite : public ConvOpLite {
 public:
  FpgaConvOpLite() {}

  explicit FpgaConvOpLite(const std::string& type) : ConvOpLite(type) {}

  bool InferShapeImpl() const override;
// TODO profile mode will be check later
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

    ch->macs = 2.f * filter_dims[2] * filter_dims[3] *
               output_dims.production() * input_dims[1] / param_.groups;
  }
#endif

  bool AttachImpl(const cpp::OpDesc& op_desc, lite::Scope* scope) override {
    ConvOpLite::AttachImpl(op_desc, scope);
#ifdef LITE_WITH_FPGA
    // additional config for fpga conv op
    auto wd_enable = op_desc.GetAttr<bool>("wd_enable");
    auto wd_offset = op_desc.GetAttr<int>("wd_offset");
    auto fuse_idx = op_desc.GetAttr<int>("fuse_idx");
    auto original_out_channel = op_desc.GetAttr<int>("original_out_channel");
    auto start_idx = op_desc.GetAttr<int>("start_idx");
    auto end_idx = op_desc.GetAttr<int>("end_idx");
    auto& stride_info_ = param_.stride_info_;
    stride_info_.wd_enable_ = wd_enable;
    stride_info_.wd_offset_ = wd_offset;
    stride_info_.fuse_idx_ = fuse_idx;
    stride_info_.original_out_channel_ = original_out_channel;
    stride_info_.start_idx_ = start_idx;
    stride_info_.end_idx_ = end_idx;
#endif
    return true;
  }

  std::string DebugString() const override { return "fpga_conv2d"; }
};

}  // namespace operators
}  // namespace lite
}  // namespace paddle
