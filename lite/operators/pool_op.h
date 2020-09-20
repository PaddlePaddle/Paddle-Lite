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

#include <algorithm>
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

class PoolOpLite : public OpLite {
 public:
  PoolOpLite() {}

  explicit PoolOpLite(const std::string &type) : OpLite(type) {}

  bool CheckShape() const override;

  bool InferShapeImpl() const override;

  // TODO(Superjomn) replace framework::OpDesc with a lite one.
  bool AttachImpl(const cpp::OpDesc &op_desc, lite::Scope *scope) override {
    AttachParam(&param_);
    auto x = op_desc.Input("X").front();
    auto out = op_desc.Output("Out").front();

    CHECK(scope->FindVar(x));
    CHECK(scope->FindVar(out));
    param_.x = scope->FindVar(x)->GetMutable<lite::Tensor>();
    param_.output = scope->FindVar(out)->GetMutable<lite::Tensor>();

    param_.pooling_type = op_desc.GetAttr<std::string>("pooling_type");
    param_.ksize = op_desc.GetAttr<std::vector<int>>("ksize");
    param_.global_pooling = op_desc.GetAttr<bool>("global_pooling");
    param_.strides = op_desc.GetAttr<std::vector<int>>("strides");
    std::vector<int> paddings = op_desc.GetAttr<std::vector<int>>("paddings");

    if (op_desc.HasAttr("exclusive")) {
      param_.exclusive = op_desc.GetAttr<bool>("exclusive");
    }
    if (op_desc.HasAttr("adaptive")) {
      param_.adaptive = op_desc.GetAttr<bool>("adaptive");
    }
    if (op_desc.HasAttr("ceil_mode")) {
      param_.ceil_mode = op_desc.GetAttr<bool>("ceil_mode");
    }
    if (op_desc.HasAttr("use_quantizer")) {
      param_.use_quantizer = op_desc.GetAttr<bool>("use_quantizer");
    }
    if (op_desc.HasAttr("padding_algorithm")) {
      padding_algorithm_ = op_desc.GetAttr<std::string>("padding_algorithm");
    }
    // 2-pad to 4-pad
    if (paddings.size() == 2L) {
      for (size_t i = 0; i < 2L; ++i) {
        int copy_pad = *(paddings.begin() + 2 * i);
        paddings.insert(paddings.begin() + 2 * i + 1, copy_pad);
      }
    } else {
      if (paddings.size() != 4L) {
        LOG(FATAL)
            << "Paddings size should be the same or twice as the inputs size.";
      }
    }
    param_.paddings = std::make_shared<std::vector<int>>(paddings);

    return true;
  }

  void AttachKernel(KernelBase *kernel) override { kernel->SetParam(param_); }

  std::string DebugString() const override { return "pool2d"; }

#ifdef LITE_WITH_PROFILE
  void GetOpRuntimeInfo(paddle::lite::profile::OpCharacter *ch) {
    auto input_dims = param_.x->dims();
    auto output_dims = param_.output->dims();
    ch->input_shape = ch->DimToStr(input_dims);
    ch->output_shape = ch->DimToStr(output_dims);
    if (param_.global_pooling) {
      ch->remark = "global" + param_.pooling_type;
    } else {
      ch->remark = param_.pooling_type + std::to_string(param_.ksize[0]) + "x" +
                   std::to_string(param_.ksize[1]) + "s" +
                   std::to_string(param_.strides[0]) + "p" +
                   std::to_string((*param_.paddings)[0]);
    }
    ch->remark += padding_algorithm_;
    ch->macs = output_dims.production() * param_.ksize[0] * param_.ksize[1];
  }
#endif

 private:
  mutable PoolParam param_;
  std::string padding_algorithm_{""};
};

inline void UpdatePadding(std::vector<int> *paddings,
                          const bool global_pooling,
                          const bool adaptive,
                          const std::string padding_algorithm,
                          const lite::DDim data_dims,
                          const std::vector<int> &strides,
                          const std::vector<int> &ksize) {
  // when padding_algorithm is "VALID" or "SAME"
  if (padding_algorithm == "SAME") {
    for (size_t i = 0; i < strides.size(); ++i) {
      int out_size = (data_dims[i + 2] + strides[i] - 1) / strides[i];
      int pad_sum =
          (std::max)((out_size - 1) * strides[i] + ksize[i] - data_dims[i + 2],
                     (int64_t)0);
      int pad_0 = pad_sum / 2;
      int pad_1 = pad_sum - pad_0;
      *(paddings->begin() + i * 2) = pad_0;
      *(paddings->begin() + i * 2 + 1) = pad_1;
    }
  } else if (padding_algorithm == "VALID") {
    for (auto it = paddings->begin(); it != paddings->end(); it++) {
      *it = 0;
    }
  }

  // if global_pooling == true or adaptive == true, padding will be ignore
  if (global_pooling || adaptive) {
    for (auto it = paddings->begin(); it != paddings->end(); it++) {
      *it = 0;
    }
  }
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle
