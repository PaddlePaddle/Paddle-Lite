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

  bool InferShape() const override;

  // TODO(Superjomn) replace framework::OpDesc with a lite one.
  bool AttachImpl(const cpp::OpDesc &op_desc, lite::Scope *scope) override {
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
    auto paddings = op_desc.GetAttr<std::vector<int>>("paddings");

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
    if (paddings.size() == param_.strides.size()) {
      for (size_t i = 0; i < param_.strides.size(); ++i) {
        int copy_pad = *(paddings.begin() + 2 * i);
        paddings.insert(paddings.begin() + 2 * i + 1, copy_pad);
      }
    } else {
      if (paddings.size() != param_.strides.size() * 2) {
        LOG(FATAL)
            << "Paddings size should be the same or twice as the inputs size.";
      }
    }
    param_.paddings = paddings;
    return true;
  }

  void AttachKernel(KernelBase *kernel) override { kernel->SetParam(param_); }

  std::string DebugString() const override { return "pool2d"; }

 private:
  mutable PoolParam param_;
  std::string padding_algorithm_{""};
};

}  // namespace operators
}  // namespace lite
}  // namespace paddle
