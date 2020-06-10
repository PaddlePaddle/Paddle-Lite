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

#include "lite/kernels/x86/interpolate_compute.h"
#include <gtest/gtest.h>
#include <iostream>
#include <memory>
#include <utility>
#include <vector>
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace x86 {

void NearestInterpRef(lite::Tensor* input,
                      lite::Tensor* output,
                      bool with_align) {
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

TEST(interpolate_x86, retrive_op) {
  auto interpolate =
      KernelRegistry::Global().Create<TARGET(kX86), PRECISION(kFloat)>(
          "nearest_interp");
  ASSERT_FALSE(interpolate.empty());
  ASSERT_TRUE(interpolate.front());
}

TEST(interpolate_x86, init) {
  InterpolateCompute interpolate;
  ASSERT_EQ(interpolate.precision(), PRECISION(kFloat));
  ASSERT_EQ(interpolate.target(), TARGET(kX86));
}

TEST(interpolate_x86, run_test) {
  lite::Tensor X, OutSize, Out, Out_base;
  operators::InterpolateParam param;
  InterpolateCompute interpolate;

  int n = 1, c = 3, in_h = 40, in_w = 40;
  int out_h = 80, out_w = 80;
  float scale = 2.0;

  param.out_h = out_h;
  param.out_w = out_w;
  param.scale = scale;
  param.align_corners = false;

  X.Resize({n, c, in_h, in_w});
  OutSize.Resize({2});
  Out.Resize({n, c, out_h, out_w});
  Out_base.Resize({n, c, out_h, out_w});

  auto* out_data = Out.mutable_data<float>();
  auto* out_base_data = Out_base.mutable_data<float>();
  auto* x_data = X.mutable_data<float>();
  auto* outsize_data = OutSize.mutable_data<float>();

  for (int i = 0; i < X.dims().production(); i++) {
    x_data[i] = i + 5.0;
  }
  outsize_data[0] = out_h;
  outsize_data[1] = out_w;

  param.X = &X;
  param.OutSize = &OutSize;
  param.Out = &Out;
  std::unique_ptr<KernelContext> ctx(new KernelContext);
  ctx->As<X86Context>();
  interpolate.SetContext(std::move(ctx));
  interpolate.SetParam(std::move(param));
  interpolate.Run();
  NearestInterpRef(&X, &Out_base, false);

  for (int i = 0; i < Out.dims().production(); i++) {
    LOG(INFO) << out_data[i];
    EXPECT_NEAR(out_data[i], out_base_data[i], 1e-5);
  }
}

}  // namespace x86
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

USE_LITE_KERNEL(nearest_interp, kX86, kFloat, kNCHW, def);
