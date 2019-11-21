/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once
#include <algorithm>
#include <vector>
#include "lite/core/op_registry.h"
#include "lite/kernels/cuda/search_grnn_compute.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace cuda {
using Tensor = lite::Tensor;

template <typename T>
T sigmoid(T z) {
  return 1 / (1 + std::exp(-z));
}

template <typename T>
__global__ void PreComputeKernel(
    const int num, const T* w_x_e, const T* wz_x_e, T* tilde, T* z, T* hidden) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < num) {
    tilde[index] = std::tanh(w_x_e[index]);
    z[index] = 1 / (1 + std::exp(-wz_x_e[index]));
    hidden[index] = (1. - z[index]) * tilde[index];
  }
}

template <typename T>
__global__ void PostComputeKernel(const int start,
                                  const int end,
                                  const int cap_h,
                                  const int w_tm1,
                                  const T* wr_x_e,
                                  const T* ur_x_h,
                                  const T* wz_x_e,
                                  const T* uz_x_h,
                                  const T* w_x_e,
                                  const T* u_x_h,
                                  T* r,
                                  T* z,
                                  T* tilde,
                                  T* hidden) {
  int j = start + blockIdx.x * blockDim.x + threadIdx.x;
  if (j < end) {
    r[j] = 1 / (1 + std::exp(-(wr_x_e[j] + ur_x_h[j])));
    z[j] = 1 / (1 + std::exp(-(wz_x_e[j] + uz_x_h[j])));
    tilde[j] = std::tanh(w_x_e[j] + r[j] * u_x_h[j]);
    hidden[j] = z[j] * hidden[j - cap_h * w_tm1] + (1.0 - z[j]) * tilde[j];
  }
}

void SearchGrnnCompute::PrepareForRun() {
  gemm_impl_.reset(new lite::cuda::math::Gemm<float, float>);
}

void SearchGrnnCompute::PrepareLayout(const Tensor* input_blob) {
  auto& param = this->Param<param_t>();
  auto& context = this->ctx_->template As<CUDAContext>();
  auto cuda_stream = context.exec_stream();

  auto* _input = input_blob;
  int dim0 = _input->dims()[0];
  int dim1 = 1;
  if (_input->dims().size() > 1) {
    dim1 = _input->dims()[1];
  }
  int batch = _input->lod()[0].size() - 1;
  auto& offset = _input->lod()[0];

  idx_sorted_by_width_cpu = std::make_shared<Tensor>();
  idx_sorted_by_width_cpu->Resize({batch});
  int* idx_sorted_by_width_cpu_data =
      idx_sorted_by_width_cpu->mutable_data<int>();

  Tensor _width;
  _width.Resize({batch});
  int* width_data = _width.mutable_data<int>();
  // sort sequence by width (descending) and find the largest width in the
  // batch
  for (int i = 0; i < batch; i++) {
    width_data[i] = offset[i + 1] - offset[i];
    idx_sorted_by_width_cpu_data[i] = i;
  }
  std::sort(idx_sorted_by_width_cpu_data,
            idx_sorted_by_width_cpu_data + batch,
            [&_width](int a, int b) {
              return _width.data<int>()[a] > _width.data<int>()[b];
            });
  int max_width = width_data[idx_sorted_by_width_cpu_data[0]];

  // start of reorganizing the input
  std::vector<size_t> new_offset;
  new_offset.resize(max_width + 1);
  new_offset[0] = 0;
  int j = batch - 1;
  int last_width = 0;
  int sub_row = 0;
  int sub_col = 0;

  for (int i = 1; i <= max_width;) {
    for (int k = j; k >= 0; --k) {
      if (width_data[idx_sorted_by_width_cpu_data[k]] > last_width) {
        sub_row = width_data[idx_sorted_by_width_cpu_data[k]] - last_width;
        sub_col = k + 1;
        for (int s = 0; s < sub_row; s++) {
          new_offset[i] = new_offset[i - 1] + sub_col;
          i++;
        }
        // move on
        last_width = width_data[idx_sorted_by_width_cpu_data[k]];
        j = k - 1;
        break;
      }
    }
  }

  // copying to the reorganized buffer
  auto* _layout_input = new Tensor();
  auto* _layout_input_gpu = param.layout_input;
  if (_input->dims().size() == 1) {
    // _layout_input.reshape_batch_sequence({dim0}, new_offset);
    LOG(FATAL) << "_input->dims().size() = 1, error.";
  } else {
    // _layout_input.reshape_batch_sequence({dim0, dim1}, new_offset);
    LoD new_lod;
    new_lod.push_back(new_offset);
    _layout_input->set_lod(new_lod);
    _layout_input->Resize({dim0, dim1});
    _layout_input_gpu->set_lod(new_lod);
    _layout_input_gpu->Resize({dim0, dim1});
  }

  auto* new_emb = _layout_input->mutable_data<float>();
  auto* input_cpu = new Tensor();
  input_cpu->Resize(_input->dims());
  auto* input_cpu_data = input_cpu->mutable_data<float>();
  TargetW::MemcpyAsync(input_cpu_data,
                       _input->data<float>(),
                       _input->numel() * sizeof(float),
                       IoDirection::DtoH,
                       cuda_stream);
  for (int i = 0; i < max_width; i++) {
    int w = new_offset[i + 1] - new_offset[i];
    auto* emb_start = new_emb + dim1 * new_offset[i];
    for (int j = 0; j < w; ++j) {
      memcpy(emb_start + dim1 * j,
             input_cpu_data + dim1 * offset[idx_sorted_by_width_cpu_data[j]] +
                 dim1 * i,
             dim1 * sizeof(float));
    }
  }

  auto* _layout_input_gpu_data =
      _layout_input_gpu->mutable_data<float>(TARGET(kCUDA));
  TargetW::MemcpyAsync(_layout_input_gpu_data,
                       new_emb,
                       _layout_input->numel() * sizeof(float),
                       IoDirection::HtoD,
                       cuda_stream);
  delete _layout_input;
  delete input_cpu;
}

void SearchGrnnCompute::CopyBack(float* from, float* to, int step) {
  auto& param = this->Param<param_t>();
  auto& context = this->ctx_->template As<CUDAContext>();
  auto stream = context.exec_stream();
  auto* _input = param.x;
  auto* _layout_input = param.layout_input;

  const auto& offset = _input->lod()[0];
  const auto& new_offset = _layout_input->lod()[0];
  const auto* idx_sorted_by_width_cpu_data =
      idx_sorted_by_width_cpu->data<int>();
  for (size_t i = 0; i < _layout_input->lod()[0].size() - 1; ++i) {
    int w = new_offset[i + 1] - new_offset[i];
    for (int j = 0; j < w; j++) {
      TargetW::MemcpyAsync(
          to + step * (offset[idx_sorted_by_width_cpu_data[j]] + i),
          from + (new_offset[i] + j) * step,
          step * sizeof(float),
          IoDirection::DtoD,
          stream);
    }
  }
}

void SearchGrnnCompute::Run() {
  CHECK(ctx_) << "running context should be set first";
  auto& param = this->Param<param_t>();
  auto& context = this->ctx_->template As<CUDAContext>();
  auto stream = context.exec_stream();

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
  auto* top_hidden = top->mutable_data<float>(TARGET(kCUDA));

  const auto* dense_e2h = wi->data<float>();
  const auto* dense_h2h = wh->data<float>();

  const auto* e2h = dense_e2h;
  const auto* e2hr = dense_e2h + 1 * _cap_e * _cap_h;
  const auto* e2hz = dense_e2h + 2 * _cap_e * _cap_h;
  const auto* h2h = dense_h2h;
  const auto* h2hr = dense_h2h + 1 * _cap_h * _cap_h;
  const auto* h2hz = dense_h2h + 2 * _cap_h * _cap_h;

  PrepareLayout(bottom);

  auto* _layout_input = param.layout_input;
  auto* new_emb = _layout_input->data<float>();
  const auto& new_offset = _layout_input->lod()[0];
  int max_width = _layout_input->lod()[0].size() - 1;

  // this buffer is used for book keeping info which will be used in bp
  // buffer also needed in bp, so make it larger
  _buffer->Resize({20, _cap_l, _cap_h});
  auto* buffer_data = _buffer->mutable_data<float>(TARGET(kCUDA));
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

  gemm_impl_->init(false, true, _cap_l, _cap_h, _cap_e, &context);
  gemm_impl_->run(1.0f, 0.0f, new_emb, e2h, w_x_e, &context);
  gemm_impl_->init(false, true, _cap_l, _cap_h, _cap_e, &context);
  gemm_impl_->run(1.0f, 0.0f, new_emb, e2hr, wr_x_e, &context);
  gemm_impl_->init(false, true, _cap_l, _cap_h, _cap_e, &context);
  gemm_impl_->run(1.0f, 0.0f, new_emb, e2hz, wz_x_e, &context);

  // precompute hidden0
  int num = batch * _cap_h;
  int threads = 512;
  int blocks = (num + threads - 1) / threads;
  PreComputeKernel<<<blocks, threads, 0, stream>>>(
      num, w_x_e, wz_x_e, tilde, z, hidden);

  // recurrence
  for (int i = 1; i < max_width; i++) {
    int w_tm1 = new_offset[i] - new_offset[i - 1];
    int w = new_offset[i + 1] - new_offset[i];

    // precompute hidden i-1 to hidden i
    auto* htm1 = hidden + new_offset[i - 1] * _cap_h;

    gemm_impl_->init(false, true, w, _cap_h, _cap_h, &context);
    gemm_impl_->run(
        1.0f, 0.0f, htm1, h2h, u_x_h + new_offset[i] * _cap_h, &context);
    gemm_impl_->init(false, true, w, _cap_h, _cap_h, &context);
    gemm_impl_->run(
        1.0f, 0.0f, htm1, h2hr, ur_x_h + new_offset[i] * _cap_h, &context);
    gemm_impl_->init(false, true, w, _cap_h, _cap_h, &context);
    gemm_impl_->run(
        1.0f, 0.0f, htm1, h2hz, uz_x_h + new_offset[i] * _cap_h, &context);

    // compute the gate and hidden
    int start = new_offset[i] * _cap_h;
    int end = (new_offset[i] + w) * _cap_h;
    PostComputeKernel<<<blocks, threads, 0, stream>>>(start,
                                                      end,
                                                      _cap_h,
                                                      w_tm1,
                                                      wr_x_e,
                                                      ur_x_h,
                                                      wz_x_e,
                                                      uz_x_h,
                                                      w_x_e,
                                                      u_x_h,
                                                      r,
                                                      z,
                                                      tilde,
                                                      hidden);
  }

  CopyBack(hidden, top_hidden, _cap_h);
}

}  // namespace cuda
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(search_grnn,
                     kCUDA,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::cuda::SearchGrnnCompute,
                     def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kCUDA),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kNCHW))})
    .BindInput("Wi",
               {LiteType::GetTensorTy(TARGET(kCUDA),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kNCHW))})
    .BindInput("Wh",
               {LiteType::GetTensorTy(TARGET(kCUDA),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kNCHW))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kCUDA),
                                       PRECISION(kFloat),
                                       DATALAYOUT(kNCHW))})
    .BindOutput("tmp_buffer",
                {LiteType::GetTensorTy(TARGET(kCUDA),
                                       PRECISION(kFloat),
                                       DATALAYOUT(kNCHW))})
    .BindOutput("idx_sorted_by_width",
                {LiteType::GetTensorTy(TARGET(kCUDA),
                                       PRECISION(kFloat),
                                       DATALAYOUT(kNCHW))})
    .BindOutput("layout_input",
                {LiteType::GetTensorTy(TARGET(kCUDA),
                                       PRECISION(kFloat),
                                       DATALAYOUT(kNCHW))})
    .Finalize();
