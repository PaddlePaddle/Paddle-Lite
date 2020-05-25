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

#include "lite/kernels/x86/match_matrix_tensor_compute.h"
#include <vector>

namespace paddle {
namespace lite {
namespace kernels {
namespace x86 {

template <typename T>
void MatchMatrixTensorCompute<T>::Run() {
  auto& context = ctx_->As<X86Context>();
  auto& param = this->Param<param_t>();
  auto* x = param.x;
  auto* w = param.w;
  auto* y = param.y;
  auto* out = param.out;
  auto* tmp = param.tmp;
  int dim_t = param.dim_t;
  int dim_in = x->dims()[1];

  const auto& offset_l = x->lod()[0];
  const auto& offset_r = y->lod()[0];

  std::vector<uint64_t> top_offset;
  int top_size = 0;
  top_offset.push_back(top_size);
  for (size_t b = 0; b < x->lod()[0].size() - 1; b++) {
    int len_l = offset_l[b + 1] - offset_l[b];
    int len_r = offset_r[b + 1] - offset_r[b];
    top_size += dim_t * len_l * len_r;
    top_offset.push_back(top_size);
  }

  auto* bottom_l_data = x->template data<T>();
  auto* bottom_r_data = y->template data<T>();
  auto* t_data = w->template data<T>();
  auto* out_data = out->template mutable_data<T>();
  auto* bottom_l_trans_data = tmp->template mutable_data<T>();
  memset(out_data, 0.0, out->dims()[0] * out->dims()[1] * sizeof(T));
  memset(bottom_l_trans_data, 0.0, tmp->dims()[0] * tmp->dims()[1] * sizeof(T));

  auto blas = lite::x86::math::GetBlas<TARGET(kX86), T>(context);
  blas.GEMM(CblasNoTrans,
            CblasNoTrans,
            x->dims()[0],
            dim_t * dim_in,
            dim_in,
            1.0f,
            bottom_l_data,
            dim_in,
            t_data,
            dim_t * dim_in,
            0.0f,
            bottom_l_trans_data,
            dim_t * dim_in);

  for (size_t b = 0; b < x->lod()[0].size() - 1; b++) {
    for (int t = 0; t < dim_t; t++) {
      int len_l = offset_l[b + 1] - offset_l[b];
      int len_r = offset_r[b + 1] - offset_r[b];
      auto* top_data = out_data + top_offset[b] + t * len_l * len_r;
      const auto* l_t_data =
          bottom_l_trans_data + offset_l[b] * dim_t * dim_in + t * dim_in;
      const auto* r_data = bottom_r_data + offset_r[b] * dim_in;

      auto blas = lite::x86::math::GetBlas<TARGET(kX86), T>(context);
      blas.GEMM(CblasNoTrans,
                CblasTrans,
                len_l,
                len_r,
                dim_in,
                1.0f,
                l_t_data,
                dim_t * dim_in,
                r_data,
                dim_in,
                0.0f,
                top_data,
                len_r);
    }
  }

  int batch_size = x->lod()[0].size() - 1;
  int lod_lv1_size = batch_size * dim_t;
  int lod_lv2_size = x->lod()[0].back() * dim_t;
  std::vector<uint64_t> out_lod0(batch_size + 1, 0);
  std::vector<uint64_t> out_lod1(lod_lv1_size + 1, 0);
  std::vector<uint64_t> out_lod2(lod_lv2_size + 1, 0);
  for (int i = 0; i < batch_size; i++) {
    out_lod0[i + 1] = out_lod0[i] + dim_t;
    int len_l = offset_l[i + 1] - offset_l[i];

    for (int j = 0; j < dim_t; j++) {
      out_lod1[i * dim_t + j + 1] = out_lod1[i * dim_t + j] + len_l;
      int len_r = offset_r[i + 1] - offset_r[i];

      for (int k = 0; k < len_l; k++) {
        out_lod2[offset_l[i] * dim_t + j * len_l + k + 1] =
            out_lod2[offset_l[i] * dim_t + j * len_l + k] + len_r;
      }
    }
  }

  LoD out_lod;
  out_lod.push_back(top_offset);
  out_lod.push_back(offset_l);
  out_lod.push_back(offset_r);
  out->set_lod(out_lod);
}

}  // namespace x86
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(
    match_matrix_tensor,
    kX86,
    kFloat,
    kNCHW,
    paddle::lite::kernels::x86::MatchMatrixTensorCompute<float>,
    def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kX86))})
    .BindInput("W", {LiteType::GetTensorTy(TARGET(kX86))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kX86))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kX86))})
    .BindOutput("Tmp", {LiteType::GetTensorTy(TARGET(kX86))})
    .Finalize();
