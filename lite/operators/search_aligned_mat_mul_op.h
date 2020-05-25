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
#include "lite/core/op_lite.h"
#include "lite/core/scope.h"
#include "lite/utils/all.h"

namespace paddle {
namespace lite {
namespace operators {

class SearchAlignedMatMulOpLite : public OpLite {
 public:
  SearchAlignedMatMulOpLite() {}

  explicit SearchAlignedMatMulOpLite(const std::string& type) : OpLite(type) {}

  bool CheckShape() const override;

  bool InferShapeImpl() const override;

  void AttachKernel(KernelBase* kernel) override { kernel->SetParam(param_); }

  bool AttachImpl(const cpp::OpDesc& op_desc, lite::Scope* scope) override;

  std::string DebugString() const override { return "search_aligned_mat_mul"; }

#ifdef LITE_WITH_PROFILE
  void GetOpRuntimeInfo(paddle::lite::profile::OpCharacter* ch) {
    ch->input_shape = ch->DimToStr(param_.X->dims());
    ch->filter_shape = ch->DimToStr(param_.Y->dims());
    ch->output_shape = ch->DimToStr(param_.Out->dims());
    ch->remark = "alpha" + std::to_string(param_.alpha) + "trans_x" +
                 std::to_string(param_.transpose_X) + "trans_y" +
                 std::to_string(param_.transpose_Y);

    const auto x_dims = param_.X->dims();
    const auto y_dims = param_.Y->dims();
    const auto& x_lod = param_.X->lod();
    const auto& y_lod = param_.Y->lod();
    const auto& x_lod_0 = x_lod[0];
    const auto& y_lod_0 = y_lod[0];

    int x_inner_size = x_dims[1];
    int y_inner_size = y_dims[1];
    int x_batch_size = x_lod_0[1];
    int y_batch_size = y_lod_0[1];
    int M = param_.transpose_X ? x_inner_size : x_batch_size;
    int N = param_.transpose_Y ? y_batch_size : y_inner_size;
    int X_K = param_.transpose_X ? x_batch_size : x_inner_size;
    int Y_K = param_.transpose_Y ? y_inner_size : y_batch_size;
    CHECK_EQ(X_K, Y_K) << "K of Input(X) and Input(Y) is not equal";
    int K = X_K;
    ch->macs = 2.0 * M * N * K;
  }
#endif

 private:
  mutable MatMulParam param_;
};

}  // namespace operators
}  // namespace lite
}  // namespace paddle
