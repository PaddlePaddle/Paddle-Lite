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

#include "lite/kernels/x86/search_grnn_compute.h"
#include <algorithm>
#include <vector>
#include "lite/backends/x86/math/blas.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace x86 {

template <typename T>
T sigmoid(T z) {
  return 1 / (1 + std::exp(-z));
}

template <typename T>
void CallGemm(const lite::x86::math::BlasT<TARGET(kX86), T>& blas,
              const CBLAS_TRANSPOSE TransA,
              const CBLAS_TRANSPOSE TransB,
              const int M,
              const int N,
              const int K,
              const T alpha,
              const T* A,
              const T* B,
              const T beta,
              T* C) {
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  blas.GEMM(TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, N);
}

template <typename T>
void SearchGrnnCompute<T>::PrepareLayout(const Tensor* input_blob) {
  auto& param = this->Param<param_t>();
  auto* _idx_sorted_by_width = param.idx_sorted_by_width;
  auto* _layout_input = param.layout_input;
  auto* _input = input_blob;

  // usually total length
  int dim0 = _input->dims()[0];
  // if it is id only sequence
  int dim1 = 1;
  // if its a embedding like sequence (dim1 would be embedding_size)
  if (_input->dims().size() > 1) {
    dim1 = _input->dims()[1];
  }

  int batch = _input->lod()[0].size() - 1;
  auto& offset = _input->lod()[0];

  Tensor _width;
  _width.Resize({batch});
  _idx_sorted_by_width->Resize({batch});
  int* width_data = _width.template mutable_data<int>();
  int* idx_sorted_by_width_data =
      _idx_sorted_by_width->template mutable_data<int>();
  // sort sequence by width (descending) and find the largest width in the
  // batch
  for (int i = 0; i < batch; i++) {
    width_data[i] = offset[i + 1] - offset[i];
    idx_sorted_by_width_data[i] = i;
  }
  std::stable_sort(idx_sorted_by_width_data,
                   idx_sorted_by_width_data + batch,
                   [&_width](int a, int b) {
                     return _width.template data<int>()[a] >
                            _width.template data<int>()[b];
                   });
  int max_width = width_data[idx_sorted_by_width_data[0]];

  // start of reorganizing the input
  std::vector<uint64_t> new_offset;
  new_offset.resize(max_width + 1);

  new_offset[0] = 0;
  int j = batch - 1;
  int last_width = 0;
  int sub_row = 0;
  int sub_col = 0;

  for (int i = 1; i <= max_width;) {
    for (int k = j; k >= 0; --k) {
      if (width_data[idx_sorted_by_width_data[k]] > last_width) {
        sub_row = width_data[idx_sorted_by_width_data[k]] - last_width;
        sub_col = k + 1;

        for (int s = 0; s < sub_row; s++) {
          new_offset[i] = new_offset[i - 1] + sub_col;
          i++;
        }
        // move on
        last_width = width_data[idx_sorted_by_width_data[k]];
        j = k - 1;
        break;
      }
    }
  }

  // copying to the reorganized buffer
  if (_input->dims().size() == 1) {
    // _layout_input.reshape_batch_sequence({dim0}, new_offset);
    LOG(FATAL) << "_input->dims().size() = 1, error.";
  } else {
    // _layout_input.reshape_batch_sequence({dim0, dim1}, new_offset);
    LoD new_lod;
    new_lod.push_back(new_offset);
    _layout_input->set_lod(new_lod);
    _layout_input->Resize({dim0, dim1});
  }

  auto* new_emb = _layout_input->template mutable_data<T>();
  for (int i = 0; i < max_width; i++) {
    int w = new_offset[i + 1] - new_offset[i];
    auto* emb_start = new_emb + dim1 * new_offset[i];
    for (int j = 0; j < w; ++j) {
      memcpy(emb_start + dim1 * j,
             _input->template data<T>() +
                 dim1 * offset[idx_sorted_by_width_data[j]] + dim1 * i,
             dim1 * sizeof(T));
    }
  }
}

template <typename T>
void SearchGrnnCompute<T>::CopyBack(T* from, T* to, int step) {
  auto& param = this->Param<param_t>();
  auto* _input = param.x;
  auto* _layout_input = param.layout_input;
  auto* _idx_sorted_by_width = param.idx_sorted_by_width;

  const auto& offset = _input->lod()[0];
  const auto& new_offset = _layout_input->lod()[0];
  const auto* idx_sorted_by_width_data =
      _idx_sorted_by_width->template data<int>();
  for (size_t i = 0; i < _layout_input->lod()[0].size() - 1; ++i) {
    int w = new_offset[i + 1] - new_offset[i];
    for (int j = 0; j < w; j++) {
      memcpy(to + step * (offset[idx_sorted_by_width_data[j]] + i),
             from + (new_offset[i] + j) * step,
             step * sizeof(T));
    }
  }
}

template <typename T>
void SearchGrnnCompute<T>::Run() {
  auto& context = ctx_->As<X86Context>();
  auto& param = this->Param<param_t>();
  auto* bottom = param.x;
  auto* wi = param.wi;
  auto* wh = param.wh;
  auto* top = param.out;
  auto* _buffer = param.tmp_buffer;
  int _cap_h = param.num_hidden;
  int _cap_e = param.num_input;

  int _cap_l = bottom->dims()[0];
  int batch = bottom->lod()[0].size() - 1;

  const auto& offset = bottom->lod()[0];
  LoD top_lod;
  top_lod.push_back(offset);
  top->set_lod(top_lod);
  std::vector<int64_t> top_dims_vec{_cap_l, _cap_h};
  top->Resize(top_dims_vec);
  auto* top_hidden = top->template mutable_data<T>();

  const auto* dense_e2h = wi->template data<T>();
  const auto* dense_h2h = wh->template data<T>();

  const auto* e2h = dense_e2h;
  const auto* e2hr = dense_e2h + 1 * _cap_e * _cap_h;
  const auto* e2hz = dense_e2h + 2 * _cap_e * _cap_h;
  const auto* h2h = dense_h2h;
  const auto* h2hr = dense_h2h + 1 * _cap_h * _cap_h;
  const auto* h2hz = dense_h2h + 2 * _cap_h * _cap_h;

  PrepareLayout(bottom);

  auto* _layout_input = param.layout_input;
  auto* new_emb = _layout_input->template mutable_data<T>();
  const auto& new_offset = _layout_input->lod()[0];
  int max_width = _layout_input->lod()[0].size() - 1;

  // this buffer is used for book keeping info which will be used in bp
  // buffer also needed in bp, so make it larger
  _buffer->Resize({20, _cap_l, _cap_h});
  auto* buffer_data = _buffer->template mutable_data<T>();
  auto* w_x_e = buffer_data + 0 * _cap_l * _cap_h;
  auto* wr_x_e = buffer_data + 1 * _cap_l * _cap_h;
  auto* wz_x_e = buffer_data + 2 * _cap_l * _cap_h;
  auto* u_x_h = buffer_data + 3 * _cap_l * _cap_h;
  auto* ur_x_h = buffer_data + 4 * _cap_l * _cap_h;
  auto* uz_x_h = buffer_data + 5 * _cap_l * _cap_h;
  auto* r = buffer_data + 6 * _cap_l * _cap_h;
  auto* z = buffer_data + 7 * _cap_l * _cap_h;
  auto* tilde = buffer_data + 8 * _cap_l * _cap_h;
  // the internal hidden
  auto* hidden = buffer_data + 19 * _cap_l * _cap_h;

  auto blas = lite::x86::math::GetBlas<TARGET(kX86), T>(context);
  CallGemm(blas,
           CblasNoTrans,
           CblasTrans,
           _cap_l,
           _cap_h,
           _cap_e,
           1.0f,
           new_emb,
           e2h,
           0.0f,
           w_x_e);
  CallGemm(blas,
           CblasNoTrans,
           CblasTrans,
           _cap_l,
           _cap_h,
           _cap_e,
           1.0f,
           new_emb,
           e2hr,
           0.0f,
           wr_x_e);
  CallGemm(blas,
           CblasNoTrans,
           CblasTrans,
           _cap_l,
           _cap_h,
           _cap_e,
           1.0f,
           new_emb,
           e2hz,
           0.0f,
           wz_x_e);

  // precompute hidden0
  for (int i = 0; i < batch * _cap_h; i++) {
    tilde[i] = std::tanh(w_x_e[i]);
    z[i] = sigmoid<T>(wz_x_e[i]);
    hidden[i] = (1. - z[i]) * tilde[i];
  }

  // recurrence
  for (int i = 1; i < max_width; i++) {
    int w_tm1 = new_offset[i] - new_offset[i - 1];
    int w = new_offset[i + 1] - new_offset[i];

    // precompute hidden i-1 to hidden i
    auto* htm1 = hidden + new_offset[i - 1] * _cap_h;

    CallGemm(blas,
             CblasNoTrans,
             CblasTrans,
             w,
             _cap_h,
             _cap_h,
             1.0f,
             htm1,
             h2h,
             0.0f,
             u_x_h + new_offset[i] * _cap_h);
    CallGemm(blas,
             CblasNoTrans,
             CblasTrans,
             w,
             _cap_h,
             _cap_h,
             1.0f,
             htm1,
             h2hr,
             0.0f,
             ur_x_h + new_offset[i] * _cap_h);
    CallGemm(blas,
             CblasNoTrans,
             CblasTrans,
             w,
             _cap_h,
             _cap_h,
             1.0f,
             htm1,
             h2hz,
             0.0f,
             uz_x_h + new_offset[i] * _cap_h);

    // compute the gate and hidden
    for (size_t j = new_offset[i] * _cap_h; j < (new_offset[i] + w) * _cap_h;
         j++) {
      r[j] = sigmoid(wr_x_e[j] + ur_x_h[j]);
      z[j] = sigmoid(wz_x_e[j] + uz_x_h[j]);
      tilde[j] = std::tanh(w_x_e[j] + r[j] * u_x_h[j]);
      hidden[j] = z[j] * hidden[j - _cap_h * w_tm1] + (1.0 - z[j]) * tilde[j];
    }
  }

  CopyBack(hidden, top_hidden, _cap_h);
}

}  // namespace x86
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(search_grnn,
                     kX86,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::x86::SearchGrnnCompute<float>,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kX86))})
    .BindInput("Wi", {LiteType::GetTensorTy(TARGET(kX86))})
    .BindInput("Wh", {LiteType::GetTensorTy(TARGET(kX86))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kX86))})
    .BindOutput("tmp_buffer", {LiteType::GetTensorTy(TARGET(kX86))})
    .BindOutput("idx_sorted_by_width",
                {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt32))})
    .BindOutput("layout_input", {LiteType::GetTensorTy(TARGET(kX86))})
    .Finalize();
