// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
#include "lite/core/op_lite.h"
#include "lite/core/scope.h"
#include "lite/utils/all.h"

namespace paddle {
namespace lite {
namespace operators {

class CosOpLite : public OpLite {
 public:
  CosOpLite() {}
  explicit CosOpLite(const std::string &op_type) : OpLite(op_type) {}

  bool CheckShape() const override;

  bool InferShape() override;

  bool AttachImpl(const cpp::OpDesc &opdesc, lite::Scope *scope) override;

  void AttachKernel(KernelBase *kernel) override { kernel->SetParam(param_); }

  std::string DebugString() const override { return "cos"; }

#ifdef LITE_WITH_PROFILE
  void GetOpRuntimeInfo(paddle::lite::profile::OpCharacter *ch) {
    auto input_dims = param_.X->dims();
    auto output_dims = param_.Out->dims();
    ch->input_shape = ch->DimToStr(input_dims);
    ch->output_shape = ch->DimToStr(output_dims);
    ch->remark = "axis" + std::to_string(param_.Axis);

    auto axis = param_.Axis;
    if (axis < 0) {
      axis += input_dims.size();
    }
    int max_num = 1;
    for (int64_t i = axis + 1; i < input_dims.size(); i++)
      max_num *= input_dims[i];
    float gops = 1.0f;
    for (int i = 1; i <= max_num; i++) gops *= i;
    ch->macs = gops * output_dims.production();
  }
#endif

 private:
  mutable CosParam param_;
};

}  // namespace operators
}  // namespace lite
}  // namespace paddle
