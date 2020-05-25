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

#include "lite/operators/search_aligned_mat_mul_op.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool SearchAlignedMatMulOpLite::CheckShape() const {
  CHECK_OR_FALSE(param_.X);
  CHECK_OR_FALSE(param_.Y);
  CHECK_OR_FALSE(param_.Out);

  return true;
}

bool SearchAlignedMatMulOpLite::InferShapeImpl() const {
  const auto x_dims = param_.X->dims();
  const auto y_dims = param_.Y->dims();
  const auto& x_lod = param_.X->lod();
  const auto& y_lod = param_.Y->lod();
  bool x_transpose = param_.transpose_X;
  bool y_transpose = param_.transpose_Y;

  CHECK_EQ(x_dims.size(), 2) << "X should be 2-D tensor";
  CHECK_EQ(y_dims.size(), 2) << "Y should be 2-D tensor";
  CHECK(!x_lod.empty()) << "The Input(X) must hold lod info.";
  CHECK(!y_lod.empty()) << "The Input(Y) must hold lod info.";

  const auto& x_lod_0 = x_lod[0];
  const auto& y_lod_0 = y_lod[0];
  CHECK_GE(x_lod_0.size(), 2) << "The Input(X)'s lod info is corrupted.";
  CHECK_GE(y_lod_0.size(), 2) << "The Input(Y)'s lod info is corrupted.";
  CHECK_EQ(x_dims[0], static_cast<int64_t>(x_lod_0.back()))
      << "The Input(X)'s lod info mismatches the actual tensor shape.";
  CHECK_EQ(y_dims[0], static_cast<int64_t>(y_lod_0.back()))
      << "The Input(Y)'s lod info mismatches the actual tensor shape.";
  CHECK_EQ(x_lod_0.size(), y_lod_0.size())
      << "The Length of X and Y must be equal.";

  int seq_num = x_lod_0.size() - 1;
  int x_inner_size = x_dims[1];
  int y_inner_size = y_dims[1];
  int x_batch_size = x_lod_0[1];
  int y_batch_size = y_lod_0[1];
  int M = x_transpose ? x_inner_size : x_batch_size;
  int N = y_transpose ? y_batch_size : y_inner_size;
  int X_K = x_transpose ? x_batch_size : x_inner_size;
  int Y_K = y_transpose ? y_inner_size : y_batch_size;
  CHECK_EQ(X_K, Y_K) << "K of Input(X) and Input(Y) is not equal";

  LoD out_lod;
  std::vector<uint64_t> out_lod_0(seq_num + 1);
  out_lod_0[0] = 0;
  for (int i = 0; i < seq_num; i++) {
    out_lod_0[i + 1] = out_lod_0[i] + M;
  }
  out_lod.push_back(out_lod_0);
  DDim out_dims(
      {static_cast<int64_t>(out_lod_0.back()), static_cast<int64_t>(N)});
  param_.Out->set_lod(out_lod);
  param_.Out->Resize(out_dims);
  return true;
}

bool SearchAlignedMatMulOpLite::AttachImpl(const cpp::OpDesc& op_desc,
                                           lite::Scope* scope) {
  CHECK(!op_desc.Input("X").empty());
  CHECK(!op_desc.Input("Y").empty());
  CHECK(!op_desc.Output("Out").empty());
  auto X = op_desc.Input("X").front();
  auto Y = op_desc.Input("Y").front();
  auto Out = op_desc.Output("Out").front();
  param_.X = GetVar<lite::Tensor>(scope, X);
  param_.Y = GetVar<lite::Tensor>(scope, Y);
  param_.Out = GetMutableVar<lite::Tensor>(scope, Out);
  param_.transpose_X = op_desc.GetAttr<bool>("transpose_X");
  param_.transpose_Y = op_desc.GetAttr<bool>("transpose_Y");
  param_.alpha = op_desc.GetAttr<float>("alpha");
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(search_aligned_mat_mul,
                 paddle::lite::operators::SearchAlignedMatMulOpLite);
