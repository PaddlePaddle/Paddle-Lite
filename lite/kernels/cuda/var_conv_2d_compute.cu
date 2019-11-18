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

#include <memory>
#include <vector>
#include "lite/backends/cuda/math/gemm.h"
#include "lite/core/op_registry.h"
#include "lite/core/target_wrapper.h"
#include "lite/core/tensor.h"
#include "lite/kernels/cuda/var_conv_2d_compute.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace cuda {

const int CUDA_NUM_THREADS = 512;

template <typename Dtype>
__global__ void var_im2col_gpu_kernel(const int n,
                                      const Dtype* data_im,
                                      const int height,
                                      const int width,
                                      const int kernel_h,
                                      const int kernel_w,
                                      const int pad_h,
                                      const int pad_w,
                                      const int stride_h,
                                      const int stride_w,
                                      const int height_col,
                                      const int width_col,
                                      Dtype* data_col) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  for (int index = idx; index < n; index += blockDim.x * gridDim.x) {
    const int h_index = index / width_col;
    const int h_col = h_index % height_col;
    const int w_col = index % width_col;
    const int c_im = h_index / height_col;
    const int c_col = c_im * kernel_h * kernel_w;
    const int h_offset = h_col * stride_h - pad_h;
    const int w_offset = w_col * stride_w - pad_w;

    Dtype* data_col_ptr = data_col;
    data_col_ptr += (c_col * height_col + h_col) * width_col + w_col;
    const Dtype* data_im_ptr = data_im;
    data_im_ptr += (c_im * height + h_offset) * width + w_offset;

    for (int i = 0; i < kernel_h; ++i) {
      for (int j = 0; j < kernel_w; ++j) {
        int h_im = h_offset + i;
        int w_im = w_offset + j;
        *data_col_ptr =
            (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width)
                ? data_im_ptr[i * width + j]
                : 0;
        data_col_ptr += height_col * width_col;
      }
    }
  }
}

void VarConv2DCompute::var_im2col(const cudaStream_t& stream) {
  auto& param = this->Param<param_t>();
  int input_channel = param.input_channel;
  int kernel_h = param.kernel_h;
  int kernel_w = param.kernel_w;
  int stride_h = param.stride_h;
  int stride_w = param.stride_w;
  // auto* in_row = param.ROW;
  // auto* in_col = param.COLUMN;
  const auto* input = param.X;
  auto* col = param.Col;

  int batch = input->lod()[0].size() - 1;
  const auto& bottom_offset = input->lod()[0];
  // 2-D lod info.
  // const auto& offset_x = in_col->lod()[0];
  // const auto& offset_y = in_row->lod()[0];
  const auto& offset_y = param.X->lod()[1];
  const auto& offset_x = param.X->lod()[2];
  // top offset is the whole size of each data sample
  std::vector<uint64_t> top_offset;
  int top_size = 0;
  top_offset.push_back(top_size);
  for (int b = 0; b < batch; ++b) {
    int width = offset_x[b + 1] - offset_x[b];
    int height = offset_y[b + 1] - offset_y[b];
    int top_im_x = 0;
    if (width == 0) {
      top_im_x = 0;
    } else {
      top_im_x = (width - 1) / stride_w + 1;
    }
    int top_im_y = 0;
    if (height == 0) {
      top_im_y = 0;
    } else {
      top_im_y = (height - 1) / stride_h + 1;
    }
    int top_x = top_im_x * top_im_y;
    int top_y = input_channel * kernel_h * kernel_w;
    top_size += top_y * top_x;
    top_offset.push_back(top_size);
  }

  LoD col_lod;
  col_lod.push_back(top_offset);
  col->set_lod(col_lod);
  std::vector<int64_t> col_dims_vec{top_size};
  col_dims_vec.push_back(1);
  col->Resize(col_dims_vec);
  auto* top_data = col->mutable_data<float>(TARGET(kCUDA));
  const auto* bottom_data = input->data<float>();

  for (int b = 0; b < batch; ++b) {
    int t_offset = top_offset[b];
    int b_offset = bottom_offset[b];
    int width = offset_x[b + 1] - offset_x[b];
    int height = offset_y[b + 1] - offset_y[b];
    if (width == 0 || height == 0) {
      continue;
    }
    int width_col = (width - 1) / stride_w + 1;
    int height_col = (height - 1) / stride_h + 1;
    const float* data_im = bottom_data + b_offset;
    float* data_col = top_data + t_offset;

    // We are going to launch channels * height_col * width_col kernels, each
    // kernel responsible for copying a single-channel grid.
    int num_kernels = height_col * width_col * input_channel;
    const int CUDA_NUM_BLOCKS =
        (num_kernels + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
    var_im2col_gpu_kernel<
        float><<<CUDA_NUM_BLOCKS, CUDA_NUM_THREADS, 0, stream>>>(
        num_kernels,
        data_im,
        height,
        width,
        kernel_h,
        kernel_w,
        ((stride_h - 1) * height + kernel_h - 1) / 2,
        ((stride_w - 1) * width + kernel_w - 1) / 2,
        stride_h,
        stride_w,
        height_col,
        width_col,
        data_col);
  }
}

void VarConv2DCompute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->template As<CUDAContext>();
  auto stream = ctx.exec_stream();

  auto* bottom = param.X;
  // auto* in_row = param.ROW;
  // auto* in_col = param.COLUMN;
  auto* w = param.W;
  auto* top = param.Out;
  auto* col = param.Col;
  int output_channel = param.output_channel;
  int input_channel = param.input_channel;
  int kernel_h = param.kernel_h;
  int kernel_w = param.kernel_w;
  int stride_h = param.stride_h;
  int stride_w = param.stride_w;

  var_im2col(stream);

  int batch = bottom->lod()[0].size() - 1;
  const auto& col_offset = col->lod()[0];
  // const auto& offset_x = in_col->lod()[0];
  // const auto& offset_y = in_row->lod()[0];
  const auto& offset_y = param.X->lod()[1];
  const auto& offset_x = param.X->lod()[2];
  std::vector<size_t> top_offset;
  std::vector<int64_t> height_vector;
  std::vector<int64_t> width_vector;
  int top_size = 0;
  top_offset.push_back(top_size);
  for (int b = 0; b < batch; ++b) {
    int width = offset_x[b + 1] - offset_x[b];
    int height = offset_y[b + 1] - offset_y[b];
    int top_im_x = 0;
    if (width == 0) {
      top_im_x = 0;
    } else {
      top_im_x = (width - 1) / stride_w + 1;
    }
    int top_im_y = 0;
    if (height == 0) {
      top_im_y = 0;
    } else {
      top_im_y = (height - 1) / stride_h + 1;
    }
    height_vector.push_back(top_im_y);
    width_vector.push_back(top_im_x);
    int top_im_size = top_im_y * top_im_x;
    top_size += output_channel * top_im_size;
    top_offset.push_back(top_size);
  }

  LoD top_lod;
  top_lod.push_back(top_offset);
  top->set_lod(top_lod);
  std::vector<int64_t> top_dims_vec{top_size};
  top_dims_vec.push_back(1);
  top->Resize(top_dims_vec);

  auto* top_data = top->mutable_data<float>(TARGET(kCUDA));
  const auto* w_data = w->data<float>();
  const auto* col_data = col->data<float>();

  std::unique_ptr<lite::cuda::math::Gemm<float, float>> gemm_impl_;
  for (int b = 0; b < batch; ++b) {
    int top_im_size = (top_offset[b + 1] - top_offset[b]) / output_channel;
    if (top_im_size == 0) {
      continue;
    }
    float* out_data = top_data + top_offset[b];
    const float* in_data = col_data + col->lod()[0][b];
    gemm_impl_.reset(new lite::cuda::math::Gemm<float, float>);
    gemm_impl_->init(false,
                     false,
                     w->dims()[0],
                     height_vector[b] * width_vector[b],
                     input_channel * kernel_h * kernel_w,
                     &ctx);
    gemm_impl_->run(1., 0., w_data, in_data, out_data, &ctx);
  }

  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) LOG(INFO) << cudaGetErrorString(error);
}

}  // namespace cuda
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(var_conv_2d,
                     kCUDA,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::cuda::VarConv2DCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kCUDA))})
    .BindInput("W", {LiteType::GetTensorTy(TARGET(kCUDA))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kCUDA))})
    .BindOutput("Col", {LiteType::GetTensorTy(TARGET(kCUDA))})
    .Finalize();
