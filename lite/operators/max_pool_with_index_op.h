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

class MaxPoolWithIndexOpLite : public OpLite {
 public:
  MaxPoolWithIndexOpLite() {}

  explicit MaxPoolWithIndexOpLite(const std::string &type) : OpLite(type) {}

  bool CheckShape() const override;

  bool InferShapeImpl() const override;

  // TODO(Superjomn) replace framework::OpDesc with a lite one.
  bool AttachImpl(const cpp::OpDesc &op_desc, lite::Scope *scope) override {
    auto x = op_desc.Input("X").front();
    auto out = op_desc.Output("Out").front();
    auto mask = op_desc.Output("Mask").front();

    CHECK(scope->FindVar(x));
    CHECK(scope->FindVar(out));
    CHECK(scope->FindVar(mask));
    param_.x = scope->FindVar(x)->GetMutable<lite::Tensor>();
    param_.output = scope->FindVar(out)->GetMutable<lite::Tensor>();

    param_.ksize = op_desc.GetAttr<std::vector<int>>("ksize");
    param_.global_pooling = op_desc.GetAttr<bool>("global_pooling");
    param_.strides = op_desc.GetAttr<std::vector<int>>("strides");
    std::vector<int> paddings = op_desc.GetAttr<std::vector<int>>("paddings");
    if (op_desc.HasAttr("adaptive")) {
      param_.adaptive = op_desc.GetAttr<bool>("adaptive");
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

  std::string DebugString() const override { return "max_pool2d_with_index"; }

 private:
  mutable PoolParam param_;
};

}  // namespace operators
}  // namespace lite
}  // namespace paddle
