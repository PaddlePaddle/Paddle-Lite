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

#include "lite/kernels/cuda/var_conv_2d_compute.h"
#include <gtest/gtest.h>
#include <memory>
#include <utility>
#include <vector>

namespace paddle {
namespace lite {
namespace kernels {
namespace cuda {

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

static void naive_sgemm(const bool transpose_A,
                        const bool transpose_B,
                        const int M,
                        const int N,
                        const int K,
                        const float alpha,
                        const float* A,  // m x k (after transpose if TransA)
                        const int lda,   // leading dimension of a
                        const float* B,  // k x n (after transpose if TransB)
                        const int ldb,   // leading dimension of b
                        const float beta,
                        float* C,  // m x n
                        const int ldc) {
  for (int m = 0; m < M; ++m) {
    for (int k = 0; k < K; ++k) {
      for (int n = 0; n < N; ++n) {
        C[m * N + n] += beta * C[m * N + n];
        size_t A_idx = 0, B_idx = 0;
        if (transpose_A) {
          A_idx = k * M + m;  // A is k x m
        } else {
          A_idx = m * K + k;  // A is m x k
        }

        if (transpose_B) {
          B_idx = n * K + k;  // B is n x k
        } else {
          B_idx = k * N + n;  // B is k x n
        }

        C[m * N + n] += alpha * A[A_idx] * B[B_idx];
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
  std::vector<size_t> top_offset;
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

  for (int b = 0; b < batch; ++b) {
    int top_im_size = (top_offset[b + 1] - top_offset[b]) / output_channel;
    if (top_im_size == 0) {
      continue;
    }

    naive_sgemm(false,
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

TEST(var_conv_2d_cuda, normal) {
  VarConv2DCompute var_conv_kernel;
  std::unique_ptr<KernelContext> ctx(new KernelContext);
  auto& context = ctx->As<CUDAContext>();

  operators::VarConv2DParam param;

  lite::Tensor X, W, ROW, COLUMN;
  lite::Tensor x_cpu, w_cpu;
  lite::Tensor Out, Col, out_cpu, col_cpu;
  int kernel_h = 5, kernel_w = 5;
  int stride_h = 1, stride_w = 1;
  int input_channel = 5, output_channel = 5;

  std::vector<int64_t> w_dims_vec;
  w_dims_vec.push_back(output_channel);
  w_dims_vec.push_back(input_channel * kernel_h * kernel_w);
  W.Resize(w_dims_vec);
  w_cpu.Resize(w_dims_vec);
  auto* w_cpu_data = w_cpu.mutable_data<float>();
  for (int i = 0; i < W.numel(); ++i) {
    w_cpu_data[i] = i - 1.f;
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
    x_lod_vec.push_back(x_lod_vec.back() + height * width);
    x_size += height * width;
  }
  for (size_t i = 0; i < x_lod_vec.size(); ++i) {
    x_lod_vec[i] *= input_channel;
  }
  x_size *= input_channel;
  std::vector<int64_t> x_dims_vec{x_size, 1};
  LoD x_lod;
  x_lod.push_back(x_lod_vec);
  x_lod.push_back(row_lod_vec);
  x_lod.push_back(column_lod_vec);
  X.Resize(x_dims_vec);
  x_cpu.Resize(x_dims_vec);
  X.set_lod(x_lod);
  x_cpu.set_lod(x_lod);
  auto* x_cpu_data = x_cpu.mutable_data<float>();
  for (int i = 0; i < X.numel(); ++i) {
    x_cpu_data[i] = i % 20 * 1.f;
  }

  int sum_num = 0;
  int out_sum_num = 0;
  for (size_t i = 0; i < row_lod_vec.size() - 1; ++i) {
    int height = row_lod_vec[i + 1] - row_lod_vec[i];
    int width = column_lod_vec[i + 1] - column_lod_vec[i];
    sum_num += height * width * input_channel * kernel_h * kernel_w;
    out_sum_num += height * width * output_channel;
  }
  col_cpu.Resize({sum_num, 1});
  out_cpu.Resize({out_sum_num, 1});
  float* out_cpu_data = out_cpu.mutable_data<float>();
  float* col_cpu_data = col_cpu.mutable_data<float>();

  X.Assign<float, lite::DDim, TARGET(kCUDA)>(x_cpu_data, x_cpu.dims());
  W.Assign<float, lite::DDim, TARGET(kCUDA)>(w_cpu_data, w_cpu.dims());

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
  var_conv_kernel.SetParam(param);
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  context.SetExecStream(stream);
  var_conv_kernel.SetContext(std::move(ctx));
  var_conv_kernel.Run();
  cudaDeviceSynchronize();

  const float* out_data = Out.data<float>();
  const float* col_data = Col.data<float>();

  CopySync<TARGET(kCUDA)>(
      out_cpu_data, out_data, sizeof(float) * Out.numel(), IoDirection::DtoH);
  CopySync<TARGET(kCUDA)>(
      col_cpu_data, col_data, sizeof(float) * Col.numel(), IoDirection::DtoH);

  lite::Tensor top_ref, col_ref;
  var_conv_2d_ref(&x_cpu,
                  &w_cpu,
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
    EXPECT_NEAR(out_cpu_data[i], top_ref.data<float>()[i], 1e-5);
  }
  for (int i = 0; i < Col.numel(); ++i) {
    EXPECT_NEAR(col_cpu_data[i], col_ref.data<float>()[i], 1e-5);
  }
}

}  // namespace cuda
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
