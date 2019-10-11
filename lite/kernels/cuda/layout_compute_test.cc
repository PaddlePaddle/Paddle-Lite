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

#include "lite/kernels/cuda/layout_compute.h"
#include <gtest/gtest.h>
#include <memory>
#include <utility>

namespace paddle {
namespace lite {
namespace kernels {
namespace cuda {

#define IN(n, c, h, w)                                 \
  input_data[w + h * input_w + c * input_h * input_w + \
             n * input_c * input_h * input_w]
#define OUT(n, c, h, w)                                    \
  output_data[w + h * output_w + c * output_h * output_w + \
              n * output_c * output_h * output_w]

template <typename Dtype>
void nchw2nhwc_ref(lite::Tensor* input, lite::Tensor* output) {
  auto* input_data = input->data<Dtype>();
  auto* output_data = output->mutable_data<Dtype>();

  int input_n = input->dims()[0];
  int input_c = input->dims()[1];
  int input_h = input->dims()[2];
  int input_w = input->dims()[3];
  int output_c = output->dims()[1];
  int output_h = output->dims()[2];
  int output_w = output->dims()[3];

  for (int n = 0; n < input_n; ++n) {
    for (int c = 0; c < input_c; ++c) {
      for (int h = 0; h < input_h; ++h) {
        for (int w = 0; w < input_w; ++w) {
          OUT(n, h, w, c) = IN(n, c, h, w);
        }
      }
    }
  }
}
#undef IN
#undef OUT

#define IN(n, h, w, c)                                 \
  input_data[c + w * input_c + h * input_w * input_c + \
             n * input_h * input_w * input_c]
#define OUT(n, h, w, c)                                    \
  output_data[c + w * output_c + h * output_w * output_c + \
              n * output_h * output_w * output_c]
template <typename Dtype>
void nhwc2nchw_ref(lite::Tensor* input, lite::Tensor* output) {
  auto* input_data = input->data<Dtype>();
  auto* output_data = output->mutable_data<Dtype>();

  int input_n = input->dims()[0];
  int input_h = input->dims()[1];
  int input_w = input->dims()[2];
  int input_c = input->dims()[3];
  int output_h = output->dims()[1];
  int output_w = output->dims()[2];
  int output_c = output->dims()[3];

  for (int n = 0; n < input_n; ++n) {
    for (int c = 0; c < input_c; ++c) {
      for (int h = 0; h < input_h; ++h) {
        for (int w = 0; w < input_w; ++w) {
          OUT(n, c, h, w) = IN(n, h, w, c);
        }
      }
    }
  }
}

template <typename Dtype>
void test_reformat(LayOutCompute<Dtype>* layout_kernel, bool nchw2nhwc) {
  std::unique_ptr<KernelContext> ctx(new KernelContext);
  auto& context = ctx->As<CUDAContext>();
  operators::LayoutParam param;

  lite::Tensor x, x_cpu, x_ref;
  lite::Tensor out, out_cpu, out_ref;
  int N = 5, C = 6, H = 7, W = 8;
  if (nchw2nhwc) {
    x.Resize({N, C, H, W});
    out.Resize({N, H, W, C});

    x_cpu.Resize({N, C, H, W});
    out_cpu.Resize({N, H, W, C});

    x_ref.Resize({N, C, H, W});
    out_ref.Resize({N, H, W, C});
  } else {
    x.Resize({N, H, W, C});
    out.Resize({N, C, H, W});

    x_cpu.Resize({N, H, W, C});
    out_cpu.Resize({N, C, H, W});

    x_ref.Resize({N, H, W, C});
    out_ref.Resize({N, C, H, W});
  }

  auto* x_cpu_data = x_cpu.mutable_data<Dtype>();
  auto* out_cpu_data = out_cpu.mutable_data<Dtype>();
  auto* x_ref_data = x_ref.mutable_data<Dtype>();

  for (int i = 0; i < x_cpu.numel(); ++i) {
    x_cpu_data[i] = static_cast<Dtype>((i + 1) % 127);
    x_ref_data[i] = static_cast<Dtype>((i + 1) % 127);
  }

  x.Assign<Dtype, lite::DDim, TARGET(kCUDA)>(x_cpu_data, x_cpu.dims());

  param.x = &x;
  param.y = &out;
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  context.SetExecStream(stream);

  layout_kernel->SetParam(param);
  layout_kernel->SetContext(std::move(ctx));
  layout_kernel->Launch();
  cudaDeviceSynchronize();
  auto* out_data = out.mutable_data<Dtype>(TARGET(kCUDA));
  CopySync<TARGET(kCUDA)>(
      out_cpu_data, out_data, sizeof(Dtype) * out.numel(), IoDirection::DtoH);
  if (nchw2nhwc) {
    nchw2nhwc_ref<Dtype>(&x_ref, &out_ref);
  } else {
    nhwc2nchw_ref<Dtype>(&x_ref, &out_ref);
  }

  auto* out_ref_data = out_ref.mutable_data<Dtype>();
  for (int i = 0; i < out.numel(); i++) {
    EXPECT_NEAR(static_cast<float>(out_cpu_data[i]),
                static_cast<float>(out_ref_data[i]),
                1e-5);
  }
}

TEST(normal, nchw2nhwc) {
  LayOutCompute<float>* layout_k = new NCHWToNHWCCompute<float>();
  test_reformat(layout_k, true);
  delete layout_k;
}

/*
TEST(normal, nhwc2nchw) {
  LayOutCompute<float> * layout_k = new NHWCToNCHWCompute<float>();
  test_reformat(layout_k, false);
  delete layout_k;
}

TEST(normal, nchw2nhwcint8) {
  LayOutCompute<int8_t> * layout_k = new NCHWToNHWCCompute<int8_t>();
  test_reformat(layout_k, true);
  delete layout_k;
}

TEST(normal, nhwc2nchwint8) {
  LayOutCompute<int8_t> * layout_k = new NHWCToNCHWCompute<int8_t>();
  test_reformat(layout_k, false);
  delete layout_k;
}
*/

}  // namespace cuda
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
