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
#include "lite/operators/op_params.h"
#include "lite/utils/all.h"

namespace paddle {
namespace lite {
namespace operators {

class MatMulV2OpLite : public OpLite {
 public:
  MatMulV2OpLite() {}

  explicit MatMulV2OpLite(const std::string &type) : OpLite(type) {}

  bool CheckShape() const override;

  bool InferShapeImpl() const override;

  bool InferShapeWithCache() const override { return true; }

  void AttachKernel(KernelBase *kernel) override { kernel->SetParam(param_); }

  bool AttachImpl(const cpp::OpDesc &op_desc, lite::Scope *scope) override;

  std::string DebugString() const override { return "matmul_v2"; }

#ifdef LITE_WITH_PROFILE
  void GetOpRuntimeInfo(paddle::lite::profile::OpCharacter *ch) {
    ch->input_shape = ch->DimToStr(param_.X->dims());
    ch->filter_shape = ch->DimToStr(param_.Y->dims());
    ch->output_shape = ch->DimToStr(param_.Out->dims());
    ch->remark = "alpha" + std::to_string(param_.alpha) + "trans_x" +
                 std::to_string(param_.transpose_X) + "trans_y" +
                 std::to_string(param_.transpose_Y);

    auto x_dims = param_.X->dims();
    auto y_dims = param_.Y->dims();
    auto m = x_dims[x_dims.size() - 2];
    auto k = x_dims[x_dims.size() - 1];
    auto n = y_dims[y_dims.size() - 1];
    if (param_.transpose_X) {
      m = x_dims[x_dims.size() - 1];
      k = x_dims[x_dims.size() - 2];
    }
    if (param_.transpose_Y) {
      n = y_dims[y_dims.size() - 2];
    }
    ch->macs = 3.f * m * n * k;
  }
#endif

 private:
  mutable MatMulParam param_;
};

}  // namespace operators
}  // namespace lite
}  // namespace paddle
