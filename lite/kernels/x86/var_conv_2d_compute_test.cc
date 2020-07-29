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

#include <gtest/gtest.h>

#include <memory>
#include <utility>
#include <vector>

#include "lite/core/op_registry.h"
#include "lite/core/tensor.h"
#include "lite/kernels/x86/var_conv_2d_compute.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace x86 {

static void im2col_ref(const lite::Tensor& input,
                       const lite::Tensor* in_row,
                       const lite::Tensor* in_col,
                       const int kernel_h,
                       const int kernel_w,
                       const int stride_h,
                       const int stride_w,
                       const int input_channel,
                       lite::Tensor* col) {
  int batch = input.lod()[0].size() - 1;
  const auto& bottom_offset = input.lod()[0];
  // 2-D lod info.
  const auto& offset_x = in_col->lod()[0];
  const auto& offset_y = in_row->lod()[0];

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
  auto* top_data = col->mutable_data<float>();
  const auto* bottom_data = input.data<float>();

  int kernel_win_size = kernel_h * kernel_w;
  int half_kernel_h = kernel_h / 2;
  int half_kernel_w = kernel_w / 2;
  for (int b = 0; b < batch; ++b) {
    int t_offset = top_offset[b];
    int b_offset = bottom_offset[b];
    int width = offset_x[b + 1] - offset_x[b];
    int height = offset_y[b + 1] - offset_y[b];
    if (width == 0 || height == 0) {
      continue;
    }
    int top_im_x = (width - 1) / stride_w + 1;
    int top_im_y = (height - 1) / stride_h + 1;
    int top_x = top_im_y * top_im_x;
    for (int z = 0; z < input_channel; ++z) {
      int row_offset = kernel_win_size * z;
      int im_offset = z * width * height;
      for (int y = 0; y < height; y += stride_h) {
        for (int x = 0; x < width; x += stride_w) {
          int col_offset = x / stride_w + y / stride_h * top_im_x;
          for (int ky = 0; ky < kernel_h; ++ky) {
            for (int kx = 0; kx < kernel_w; ++kx) {
              int im_y = y + ky - half_kernel_h;
              int im_x = x + kx - half_kernel_w;
              if (im_x >= 0 && im_x < width && im_y >= 0 && im_y < height) {
                top_data[t_offset + (row_offset + ky * kernel_w + kx) * top_x +
                         col_offset] =
                    bottom_data[b_offset + im_offset + im_y * width + im_x];
              } else {
                top_data[t_offset + (row_offset + ky * kernel_w + kx) * top_x +
                         col_offset] = 0;
              }
            }
          }
        }
      }
    }
  }
}

static void var_conv_2d_ref(const lite::Tensor* bottom,
                            const lite::Tensor* w,
                            const lite::Tensor* in_row,
                            const lite::Tensor* in_col,
                            const int kernel_h,
                            const int kernel_w,
                            const int stride_h,
                            const int stride_w,
                            const int input_channel,
                            const int output_channel,
                            lite::Tensor* top,
                            lite::Tensor* col) {
  std::unique_ptr<KernelContext> ctx(new KernelContext);
  auto& context = ctx->As<X86Context>();

  im2col_ref(*bottom,
             in_row,
             in_col,
             kernel_h,
             kernel_w,
             stride_h,
             stride_w,
             input_channel,
             col);
  int batch = bottom->lod()[0].size() - 1;
  const auto& col_offset = col->lod()[0];
  const auto& offset_x = in_col->lod()[0];
  const auto& offset_y = in_row->lod()[0];
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
  auto* top_data = top->mutable_data<float>();
  const auto* w_data = w->data<float>();
  const auto* col_data = col->data<float>();

  auto blas = lite::x86::math::GetBlas<lite::TargetType::kX86, float>(context);
  for (int b = 0; b < batch; ++b) {
    int top_im_size = (top_offset[b + 1] - top_offset[b]) / output_channel;
    if (top_im_size == 0) {
      continue;
    }

    blas.GEMM(false,
              false,
              output_channel,
              top_im_size,
              input_channel * kernel_h * kernel_w,
              1.0,
              w_data,
              input_channel * kernel_h * kernel_w,
              col_data + col_offset[b],
              top_im_size,
              0.0,
              top_data + top_offset[b],
              top_im_size);
  }
}

TEST(var_conv_2d_x86, retrive_op) {
  auto var_conv_2d = KernelRegistry::Global().Create("var_conv_2d");
  ASSERT_FALSE(var_conv_2d.empty());
  ASSERT_TRUE(var_conv_2d.front());
}

TEST(var_conv_2d_x86, init) {
  VarConv2DCompute<float> var_conv_2d;
  ASSERT_EQ(var_conv_2d.precision(), PRECISION(kFloat));
  ASSERT_EQ(var_conv_2d.target(), TARGET(kX86));
}

TEST(var_conv_2d_x86, run_test) {
  VarConv2DCompute<float> var_conv_2d;
  std::unique_ptr<KernelContext> ctx(new KernelContext);
  ctx->As<X86Context>();

  operators::VarConv2DParam param;

  lite::Tensor X, W, ROW, COLUMN;
  lite::Tensor Out, Col;
  int kernel_h, kernel_w;
  int stride_h, stride_w;
  int input_channel, output_channel;

  output_channel = 5;
  input_channel = 5;
  kernel_h = 5;
  kernel_w = 5;
  stride_h = 1;
  stride_w = 1;
  std::vector<int64_t> w_dims_vec;
  w_dims_vec.push_back(output_channel);
  w_dims_vec.push_back(input_channel * kernel_h * kernel_w);
  W.Resize(w_dims_vec);
  auto* w_data = W.mutable_data<float>();
  for (int i = 0; i < W.numel(); ++i) {
    w_data[i] = i - 1.f;
  }

  std::vector<uint64_t> row_lod_vec{0, 10, 20};
  LoD row_lod;
  row_lod.push_back(row_lod_vec);
  ROW.set_lod(row_lod);

  std::vector<uint64_t> column_lod_vec{0, 10, 20};
  LoD column_lod;
  column_lod.push_back(column_lod_vec);
  COLUMN.set_lod(column_lod);

  int x_size = 0;
  std::vector<uint64_t> x_lod_vec;
  x_lod_vec.push_back(0);
  for (size_t i = 0; i < row_lod_vec.size() - 1; ++i) {
    int height = row_lod_vec[i + 1] - row_lod_vec[i];
    int width = column_lod_vec[i + 1] - column_lod_vec[i];
    x_lod_vec.push_back(height * width * input_channel);
    x_size += height * width * input_channel;
  }
  std::vector<int64_t> x_dims_vec{x_size, 1};
  LoD x_lod;
  x_lod.push_back(x_lod_vec);
  x_lod.push_back(row_lod_vec);
  x_lod.push_back(column_lod_vec);
  X.Resize(x_dims_vec);
  X.set_lod(x_lod);
  auto* x_data = X.mutable_data<float>();
  for (int i = 0; i < X.numel(); ++i) {
    x_data[i] = i % 20 * 1.f;
  }

  param.X = &X;
  param.W = &W;
  // param.ROW = &ROW;
  // param.COLUMN = &COLUMN;
  param.Out = &Out;
  param.Col = &Col;
  param.stride_h = stride_h;
  param.stride_w = stride_w;
  param.kernel_h = kernel_h;
  param.kernel_w = kernel_w;
  param.input_channel = input_channel;
  param.output_channel = output_channel;
  var_conv_2d.SetParam(param);
  var_conv_2d.SetContext(std::move(ctx));
  var_conv_2d.Run();

  lite::Tensor top_ref, col_ref;
  var_conv_2d_ref(&X,
                  &W,
                  &ROW,
                  &COLUMN,
                  kernel_h,
                  kernel_w,
                  stride_h,
                  stride_w,
                  input_channel,
                  output_channel,
                  &top_ref,
                  &col_ref);

  for (int i = 0; i < Out.numel(); ++i) {
    EXPECT_NEAR(Out.data<float>()[i], top_ref.data<float>()[i], 1e-5);
  }
  for (int i = 0; i < Col.numel(); ++i) {
    EXPECT_NEAR(Col.data<float>()[i], col_ref.data<float>()[i], 1e-5);
  }
}

}  // namespace x86
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

USE_LITE_KERNEL(var_conv_2d, kX86, kFloat, kNCHW, def);
