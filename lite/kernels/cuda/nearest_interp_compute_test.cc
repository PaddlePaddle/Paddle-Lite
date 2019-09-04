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
#include "lite/fluid/eigen.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace cuda {

template <typename T,
          size_t D,
          int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenTensor = lite::fluid::EigenTensor<T, D, MajorType, IndexType>;
using Tensor = lite::Tensor;

static void NearestNeighborInterpolate(const Tensor& input,
                                       Tensor* output,
                                       const float ratio_h,
                                       const float ratio_w,
                                       const int n,
                                       const int c,
                                       const int out_h,
                                       const int out_w,
                                       const bool align_corners) {
  auto input_t = EigenTensor<float, 4>::From(input);
  auto output_t = EigenTensor<float, 4>::From(*output);
  for (int k = 0; k < out_h; k++) {  // loop for images
    int in_k = (align_corners) ? static_cast<int>(ratio_h * k + 0.5)
                               : static_cast<int>(ratio_h * k);
    for (int l = 0; l < out_w; l++) {
      int in_l = (align_corners) ? static_cast<int>(ratio_w * l + 0.5)
                                 : static_cast<int>(ratio_w * l);
      for (int i = 0; i < n; i++) {    // loop for batches
        for (int j = 0; j < c; j++) {  // loop for channels
          output_t(i, j, k, l) = input_t(i, j, in_k, in_l);
        }
      }
    }
  }
}

static void NearestInterpRef(operators::InterpolateParam param,
                             Tensor* input,
                             const size_t scale,
                             const size_t n,
                             const size_t c,
                             const size_t in_h,
                             const size_t in_w,
                             Tensor* output_size,
                             Tensor* output,
                             size_t out_h,
                             size_t out_w) {
  if (scale > 0) {
    out_h = static_cast<int>(in_h * scale);
    out_w = static_cast<int>(in_w * scale);
  }
  bool align_corners = param.align_corners;
  if (output_size != nullptr) {
    auto out_size_data = output_size->mutable_data<float>();
    out_h = static_cast<int>(out_size_data[0]);
    out_w = static_cast<int>(out_size_data[1]);
  }

  float* input_data = input->mutable_data<float>();
  LOG(INFO) << *(input_data + 2);
  float* output_data = output->mutable_data<float>();
  LOG(INFO) << *(output_data + 2);
  if (in_h == out_h && in_w == out_w) {
    std::memcpy(output_data, input_data, sizeof(float) * n * c * in_h * in_w);
    LOG(INFO) << *(output_data + 2);
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
  NearestNeighborInterpolate(
      *input, output, ratio_h, ratio_w, n, c, out_h, out_w, align_corners);
}

TEST(nearest_interp, normal) {
  NearestInterpCompute nearest_interp_kernel;
  std::unique_ptr<KernelContext> ctx(new KernelContext);
  auto& context = ctx->As<CUDAContext>();

  operators::InterpolateParam param;

  Tensor x, osz, out;
  Tensor x_cpu, osz_cpu, out_cpu;
  Tensor x_ref, osz_ref, out_ref;

  int n = 1, c = 3, in_h = 4, in_w = 4;
  int in_chw = c * in_h * in_w;
  int out_h = 4, out_w = 4;
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

  auto* x_data = x.mutable_data<float>(TARGET(kCUDA));
  auto* osz_data = osz.mutable_data<float>(TARGET(kCUDA));
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
  NearestInterpRef(
      param, &x_ref, scale, n, c, in_h, in_w, &osz_ref, &out_ref, out_h, out_w);
  for (int i = 0; i < out.numel(); i++) {
    EXPECT_NEAR(out_cpu_data[i], out_ref_data[i], 1e-5);
  }
}

}  // namespace cuda
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
