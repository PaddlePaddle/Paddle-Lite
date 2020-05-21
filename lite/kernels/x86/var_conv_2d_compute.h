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
#pragma once

#include <vector>
#include "lite/backends/x86/math/blas.h"
#include "lite/core/kernel.h"
#include "lite/core/op_registry.h"
#include "lite/core/tensor.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace x86 {

template <typename T>
class VarConv2DCompute : public KernelLite<TARGET(kX86), PRECISION(kFloat)> {
 public:
  using param_t = operators::VarConv2DParam;

  void Im2Col(const lite::Tensor& input, lite::Tensor* col) const {
    auto& param = *param_.get_mutable<param_t>();
    int input_channel = param.input_channel;
    int kernel_h = param.kernel_h;
    int kernel_w = param.kernel_w;
    int stride_h = param.stride_h;
    int stride_w = param.stride_w;
    // auto* in_row = param.ROW;
    // auto* in_col = param.COLUMN;

    int batch = input.lod()[0].size() - 1;
    const auto& bottom_offset = input.lod()[0];
    // 2-D lod info.
    // const auto& offset_x = in_col->lod()[0];
    // const auto& offset_y = in_row->lod()[0];
    CHECK_EQ(param.X->lod().size(), 3u) << "input lod size should be 3!";
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
    // std::vector<int64_t> col_lod_vec;
    // col_lod_vec.push_back(top_offset);
    LoD col_lod;
    col_lod.push_back(top_offset);
    col->set_lod(col_lod);
    std::vector<int64_t> col_dims_vec{top_size};
    col_dims_vec.push_back(1);
    col->Resize(col_dims_vec);
    auto* top_data = col->template mutable_data<T>();
    const auto* bottom_data = input.data<T>();

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
                  top_data[t_offset +
                           (row_offset + ky * kernel_w + kx) * top_x +
                           col_offset] =
                      bottom_data[b_offset + im_offset + im_y * width + im_x];
                } else {
                  top_data[t_offset +
                           (row_offset + ky * kernel_w + kx) * top_x +
                           col_offset] = 0;
                }
              }
            }
          }
        }
      }
    }
  }

  void Run() override {
    auto& param = *param_.get_mutable<param_t>();
    auto& context = ctx_->As<X86Context>();
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

    Im2Col(*bottom, col);
    int batch = bottom->lod()[0].size() - 1;
    const auto& col_offset = col->lod()[0];
    // const auto& offset_x = in_col->lod()[0];
    // const auto& offset_y = in_row->lod()[0];
    const auto& offset_y = param.X->lod()[1];
    const auto& offset_x = param.X->lod()[2];
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
    auto* top_data = top->template mutable_data<T>();
    const auto* w_data = w->template data<T>();
    const auto* col_data = col->template data<T>();

    auto blas = lite::x86::math::GetBlas<lite::TargetType::kX86, T>(context);
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

  virtual ~VarConv2DCompute() = default;
};

}  // namespace x86
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
