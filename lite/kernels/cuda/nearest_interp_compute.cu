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
#include <vector>
#include "lite/backends/cuda/target_wrapper.h"
#include "lite/core/op_registry.h"
#include "lite/kernels/cuda/nearest_interp_compute.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace cuda {
using Tensor = lite::Tensor;

inline std::vector<int> get_new_shape(
    std::vector<const lite::Tensor*> list_new_shape_tensor) {
  // get tensor from
  std::vector<int> vec_new_shape;
  for (size_t i = 0; i < list_new_shape_tensor.size(); ++i) {
    auto tensor = list_new_shape_tensor[i];
    lite::Tensor temp;
    auto temp_data = temp.mutable_data<float>();
    auto tensor_data = tensor->data<float>();
    cudaMemcpy(temp_data,
               tensor_data,
               tensor->dims().production() * sizeof(float),
               cudaMemcpyDeviceToHost);

    vec_new_shape.push_back(static_cast<int32_t>(*temp_data));
  }

  return vec_new_shape;
}

template <typename T>
inline std::vector<T> get_new_data_from_tensor(const Tensor* new_data_tensor) {
  std::vector<T> vec_new_data;
  auto* new_data = new_data_tensor->data<T>();
  lite::Tensor cpu_starts_tensor;
  auto cpu_starts_tensor_data = cpu_starts_tensor.mutable_data<T>();
  cudaMemcpy(cpu_starts_tensor_data,
             new_data,
             new_data_tensor->dims().production() * sizeof(T),
             cudaMemcpyDeviceToHost);

  auto new_data_ = cpu_starts_tensor.data<T>();
  vec_new_data = std::vector<T>(
      new_data_, new_data_ + new_data_tensor->dims().production());
  return vec_new_data;
}

__global__ void KeNearestNeighborInterp(const float* in,
                                        const size_t in_img_h,
                                        const size_t in_img_w,
                                        const size_t input_h,
                                        const size_t input_w,
                                        float* out,
                                        const size_t out_img_h,
                                        const size_t out_img_w,
                                        const size_t output_h,
                                        const size_t output_w,
                                        const size_t num_channels,
                                        const float ratio_h,
                                        const float ratio_w,
                                        const bool align_corners) {
  int nthreads = output_h * output_w;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (; tid < nthreads; tid += stride) {
    int out_id_h = tid / output_w;
    int out_id_w = tid % output_w;
    int in_img_size = input_w / num_channels;
    int out_img_size = output_w / num_channels;
    int channel_id = out_id_w / out_img_size;

    int out_img_idy = (out_id_w % out_img_size) / out_img_w;
    int in_img_idy = (align_corners)
                         ? static_cast<int>(ratio_h * out_img_idy + 0.5)
                         : static_cast<int>(ratio_h * out_img_idy);

    int out_img_idx = tid % out_img_w;
    int in_img_idx = (align_corners)
                         ? static_cast<int>(ratio_w * out_img_idx + 0.5)
                         : static_cast<int>(ratio_w * out_img_idx);

    out[tid] = in[out_id_h * input_w + channel_id * in_img_size +
                  in_img_idy * in_img_w + in_img_idx];
  }
}

void NearestInterpCompute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->template As<CUDAContext>();
  auto stream = ctx.exec_stream();

  Tensor* input = param.X;
  Tensor* output = param.Out;
  Tensor* out_size = param.OutSize;

  auto* input_data = input->data<float>();

  const int n = input->dims()[0];
  const int c = input->dims()[1];
  const int in_h = input->dims()[2];
  const int in_w = input->dims()[3];

  int out_h = param.out_h;
  int out_w = param.out_w;
  float scale = param.scale;
  bool align_corners = param.align_corners;
  auto align_mode = param.align_mode;

  auto list_new_shape_tensor = param.SizeTensor;
  if (list_new_shape_tensor.size() > 0) {
    // have size tensor
    auto new_size = get_new_shape(list_new_shape_tensor);
    out_h = new_size[0];
    out_w = new_size[1];
  } else {
    auto scale_tensor = param.Scale;
    if (scale_tensor != nullptr) {
      auto scale_data = get_new_data_from_tensor<float>(scale_tensor);
      scale = scale_data[0];
    }
    if (scale > 0) {
      out_h = static_cast<int>(in_h * scale);
      out_w = static_cast<int>(in_w * scale);
    }

    if (out_size != nullptr) {
      lite::Tensor sizes;
      float* size_data = sizes.mutable_data<float>();
      float* outsize_data = out_size->mutable_data<float>(TARGET(kCUDA));
      cudaMemcpy(
          size_data, outsize_data, sizeof(float) * 2, cudaMemcpyDeviceToHost);
      out_h = static_cast<int>(size_data[0]);
      out_w = static_cast<int>(size_data[1]);
    }
  }

  auto output_data = output->mutable_data<float>(TARGET(kCUDA));

  if (in_h == out_h && in_w == out_w) {
    cudaMemcpy(output_data,
               input_data,
               sizeof(float) * n * c * in_h * in_w,
               cudaMemcpyHostToDevice);
    return;
  }

  float ratio_h = 0.f;
  float ratio_w = 0.f;
  if (out_h > 1) {
    ratio_h = (align_corners) ? static_cast<float>(in_h - 1) / (out_h - 1)
                              : static_cast<float>(in_h) / out_h;
  }
  if (out_w > 1) {
    ratio_w = (align_corners) ? static_cast<float>(in_w - 1) / (out_w - 1)
                              : static_cast<float>(in_w) / out_w;
  }

  int in_hw = in_h * in_w;
  int out_hw = out_h * out_w;
  int in_chw = c * in_hw;
  int out_chw = c * out_hw;

  int pixel_num = n * out_chw;
  int threads = 512;
  int blocks = (pixel_num + threads - 1) / threads;
  blocks = blocks > 8 ? 8 : blocks;

  KeNearestNeighborInterp<<<blocks, threads, 0, stream>>>(input_data,
                                                          in_h,
                                                          in_w,
                                                          n,
                                                          in_chw,
                                                          output_data,
                                                          out_h,
                                                          out_w,
                                                          n,
                                                          out_chw,
                                                          c,
                                                          ratio_h,
                                                          ratio_w,
                                                          align_corners);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) LOG(INFO) << cudaGetErrorString(error);
}

}  // namespace cuda
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(nearest_interp,
                     kCUDA,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::cuda::NearestInterpCompute,
                     def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kCUDA),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kNCHW))})
    .BindInput("OutSize",
               {LiteType::GetTensorTy(TARGET(kCUDA),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kNCHW))})
    .BindInput("SizeTensor",
               {LiteType::GetTensorTy(TARGET(kCUDA),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kNCHW))})
    .BindInput("Scale",
               {LiteType::GetTensorTy(TARGET(kCUDA),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kNCHW))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kCUDA),
                                       PRECISION(kFloat),
                                       DATALAYOUT(kNCHW))})
    .Finalize();
