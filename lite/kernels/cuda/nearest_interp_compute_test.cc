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

#include "lite/kernels/cuda/nearest_interp_compute.h"
#include <gtest/gtest.h>
#include <memory>
#include <utility>
#include <vector>

namespace paddle {
namespace lite {
namespace kernels {
namespace cuda {

using Tensor = lite::Tensor;

void NearestInterpRef(Tensor* input, Tensor* output, bool with_align) {
  int hin = input->dims()[2];
  int win = input->dims()[3];
  int channels = input->dims()[1];
  int num = input->dims()[0];
  int hout = output->dims()[2];
  int wout = output->dims()[3];
  float scale_w = (with_align) ? (static_cast<float>(win - 1) / (wout - 1))
                               : (static_cast<float>(win) / (wout));
  float scale_h = (with_align) ? (static_cast<float>(hin - 1) / (hout - 1))
                               : (static_cast<float>(hin) / (hout));
  const float* src = input->data<float>();
  float* dst = output->mutable_data<float>();
  int dst_stride_w = 1;
  int dst_stride_h = wout;
  int dst_stride_c = wout * hout;
  int dst_stride_batch = wout * hout * channels;
  int src_stride_w = 1;
  int src_stride_h = win;
  int src_stride_c = win * hin;
  int src_stride_batch = win * hin * channels;
  for (int n = 0; n < num; ++n) {
    for (int c = 0; c < channels; ++c) {
      int src_index = n * src_stride_batch + c * src_stride_c;
      for (int h = 0; h < hout; ++h) {
        for (int w = 0; w < wout; ++w) {
          int fw = (with_align) ? static_cast<int>(scale_w * w + 0.5)
                                : static_cast<int>(scale_w * w);
          fw = (fw < 0) ? 0 : fw;
          int fh = (with_align) ? static_cast<int>(scale_h * h + 0.5)
                                : static_cast<int>(scale_h * h);
          fh = (fh < 0) ? 0 : fh;
          int w_start = static_cast<int>(fw);
          int h_start = static_cast<int>(fh);
          int dst_index = n * dst_stride_batch + c * dst_stride_c +
                          h * dst_stride_h + w * dst_stride_w;
          dst[dst_index] =
              src[src_index + w_start * src_stride_w + h_start * src_stride_h];
        }
      }
    }
  }
}

TEST(nearest_interp, normal) {
  NearestInterpCompute nearest_interp_kernel;
  std::unique_ptr<KernelContext> ctx(new KernelContext);
  auto& context = ctx->As<CUDAContext>();

  operators::InterpolateParam param;

  Tensor x, osz, out;
  Tensor x_cpu, osz_cpu, out_cpu;
  Tensor x_ref, osz_ref, out_ref;

  int n = 1, c = 3, in_h = 40, in_w = 40;
  int out_h = 80, out_w = 80;
  float scale = 2.0;

  param.out_h = out_h;
  param.out_w = out_w;
  param.scale = scale;
  param.align_corners = false;

  x.Resize({n, c, in_h, in_w});
  osz.Resize({2});
  out.Resize({n, c, out_h, out_w});

  x_cpu.Resize({n, c, in_h, in_w});
  osz_cpu.Resize({2});
  out_cpu.Resize({n, c, out_h, out_w});

  x_ref.Resize({n, c, in_h, in_w});
  osz_ref.Resize({2});
  out_ref.Resize({n, c, out_h, out_w});

  auto* out_data = out.mutable_data<float>(TARGET(kCUDA));

  float* x_cpu_data = x_cpu.mutable_data<float>();
  float* osz_cpu_data = osz_cpu.mutable_data<float>();
  float* out_cpu_data = out_cpu.mutable_data<float>();

  float* x_ref_data = x_ref.mutable_data<float>();
  float* osz_ref_data = osz_ref.mutable_data<float>();
  float* out_ref_data = out_ref.mutable_data<float>();

  for (int i = 0; i < x_cpu.numel(); ++i) {
    x_cpu_data[i] = i + 5.0;
    x_ref_data[i] = i + 5.0;
  }
  osz_cpu_data[0] = out_h;
  osz_cpu_data[1] = out_w;
  osz_ref_data[0] = out_h;
  osz_ref_data[1] = out_w;

  x.Assign<float, lite::DDim, TARGET(kCUDA)>(x_cpu_data, x_cpu.dims());
  osz.Assign<float, lite::DDim, TARGET(kCUDA)>(osz_cpu_data, osz_cpu.dims());

  param.X = &x;
  param.OutSize = &osz;
  param.Out = &out;
  nearest_interp_kernel.SetParam(param);

  cudaStream_t stream;
  cudaStreamCreate(&stream);
  context.SetExecStream(stream);

  nearest_interp_kernel.SetContext(std::move(ctx));
  nearest_interp_kernel.Launch();
  cudaDeviceSynchronize();

  CopySync<TARGET(kCUDA)>(
      out_cpu_data, out_data, sizeof(float) * out.numel(), IoDirection::DtoH);
  NearestInterpRef(&x_ref, &out_ref, false);
  for (int i = 0; i < out.numel(); i++) {
    EXPECT_NEAR(out_cpu_data[i], out_ref_data[i], 1e-5);
  }
}

TEST(nearest_interp, update) {
  NearestInterpCompute nearest_interp_kernel;
  std::unique_ptr<KernelContext> ctx(new KernelContext);
  auto& context = ctx->As<CUDAContext>();

  operators::InterpolateParam param;

  std::vector<Tensor> size_tensor(2);
  std::vector<Tensor> size_tensor_cpu(2), size_tensor_ref(2);
  Tensor x, input_scale, osz, out;
  Tensor x_cpu, input_scale_cpu, osz_cpu, out_cpu;
  Tensor x_ref, input_scale_ref, osz_ref, out_ref;

  int n = 1, c = 3, in_h = 40, in_w = 40;
  int out_h = 80, out_w = 80;
  float scale = 2.0;

  param.out_h = out_h;
  param.out_w = out_w;
  param.scale = scale;
  param.align_corners = false;
  param.align_mode = 0;

  x.Resize({n, c, in_h, in_w});
  size_tensor[0].Resize({1});
  size_tensor[1].Resize({1});
  input_scale.Resize({1});
  osz.Resize({2});
  out.Resize({n, c, out_h, out_w});

  x_cpu.Resize({n, c, in_h, in_w});
  size_tensor_cpu[0].Resize({1});
  size_tensor_cpu[1].Resize({1});
  input_scale_cpu.Resize({1});
  osz_cpu.Resize({2});
  out_cpu.Resize({n, c, out_h, out_w});

  x_ref.Resize({n, c, in_h, in_w});
  size_tensor_ref[0].Resize({1});
  size_tensor_ref[1].Resize({1});
  input_scale_ref.Resize({1});
  osz_ref.Resize({2});
  out_ref.Resize({n, c, out_h, out_w});

  auto* out_data = out.mutable_data<float>(TARGET(kCUDA));

  float* x_cpu_data = x_cpu.mutable_data<float>();
  float* size_tensor0_cpu_data = size_tensor_cpu[0].mutable_data<float>();
  float* size_tensor1_cpu_data = size_tensor_cpu[1].mutable_data<float>();
  float* input_scale_cpu_data = input_scale_cpu.mutable_data<float>();
  float* osz_cpu_data = osz_cpu.mutable_data<float>();
  float* out_cpu_data = out_cpu.mutable_data<float>();

  float* x_ref_data = x_ref.mutable_data<float>();
  float* size_tensor0_ref_data = size_tensor_ref[0].mutable_data<float>();
  float* size_tensor1_ref_data = size_tensor_ref[1].mutable_data<float>();
  float* input_scale_ref_data = input_scale_ref.mutable_data<float>();
  float* osz_ref_data = osz_ref.mutable_data<float>();
  float* out_ref_data = out_ref.mutable_data<float>();

  for (int i = 0; i < x_cpu.numel(); ++i) {
    x_cpu_data[i] = i + 5.0;
    x_ref_data[i] = i + 5.0;
  }
  osz_cpu_data[0] = out_h;
  osz_cpu_data[1] = out_w;
  size_tensor0_cpu_data[0] = out_h;
  size_tensor1_cpu_data[0] = out_w;
  input_scale_cpu_data[0] = scale;
  osz_ref_data[0] = out_h;
  osz_ref_data[1] = out_w;
  size_tensor0_ref_data[0] = out_h;
  size_tensor1_ref_data[0] = out_w;
  input_scale_ref_data[0] = scale;

  x.Assign<float, lite::DDim, TARGET(kCUDA)>(x_cpu_data, x_cpu.dims());
  size_tensor[0].Assign<float, lite::DDim, TARGET(kCUDA)>(
      size_tensor0_cpu_data, size_tensor[0].dims());
  size_tensor[1].Assign<float, lite::DDim, TARGET(kCUDA)>(
      size_tensor1_cpu_data, size_tensor[1].dims());
  input_scale.Assign<float, lite::DDim, TARGET(kCUDA)>(input_scale_cpu_data,
                                                       input_scale.dims());
  osz.Assign<float, lite::DDim, TARGET(kCUDA)>(osz_cpu_data, osz_cpu.dims());

  param.X = &x;
  param.SizeTensor.emplace_back(
      reinterpret_cast<const Tensor*>(&size_tensor[0]));
  param.SizeTensor.emplace_back(
      reinterpret_cast<const Tensor*>(&size_tensor[1]));
  param.Scale = &input_scale;
  param.OutSize = &osz;
  param.Out = &out;
  nearest_interp_kernel.SetParam(param);

  cudaStream_t stream;
  cudaStreamCreate(&stream);
  context.SetExecStream(stream);

  nearest_interp_kernel.SetContext(std::move(ctx));
  nearest_interp_kernel.Launch();
  cudaDeviceSynchronize();

  CopySync<TARGET(kCUDA)>(
      out_cpu_data, out_data, sizeof(float) * out.numel(), IoDirection::DtoH);
  NearestInterpRef(&x_ref, &out_ref, false);
  for (int i = 0; i < out.numel(); i++) {
    EXPECT_NEAR(out_cpu_data[i], out_ref_data[i], 1e-5);
  }
}

}  // namespace cuda
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
