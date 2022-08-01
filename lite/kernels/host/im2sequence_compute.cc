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

#include "lite/kernels/host/im2sequence_compute.h"
#include <vector>
#include "lite/api/paddle_place.h"
#include "lite/core/op_registry.h"
#include "lite/core/type_system.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace host {

void im2sequence(const float* input,
                 const int input_c,
                 const int input_h,
                 const int input_w,
                 const int kernel_h,
                 const int kernel_w,
                 const int pad_top,
                 const int pad_bottom,
                 const int pad_left,
                 const int pad_right,
                 const int stride_h,
                 const int stride_w,
                 const int out_h,
                 const int out_w,
                 float* out,
                 Context<TARGET(kHost)>* ctx) {
  int im_channels = input_c;
  int im_height = input_h;
  int im_width = input_w;
  int filter_height = kernel_h;
  int filter_width = kernel_w;
  int output_height = out_h;
  int output_width = out_w;
  int channels_col = im_channels * filter_height * filter_width;

  for (int c = 0; c < channels_col; ++c) {
    int w_offset = c % filter_width;
    int h_offset = (c / filter_width) % filter_height;
    int c_im = c / (filter_width * filter_height);
    for (int h = 0; h < output_height; ++h) {
      int im_row_idx = h * stride_h - pad_top + h_offset;
      for (int w = 0; w < output_width; ++w) {
        int im_col_idx = w * stride_w - pad_left + w_offset;
        int im_idx = (im_row_idx + c_im * im_height) * im_width + im_col_idx;
        int col_idx = h * output_width * channels_col + w * channels_col + c;
        out[col_idx] = (im_row_idx < 0 || im_row_idx >= im_height ||
                        im_col_idx < 0 || im_col_idx >= im_width)
                           ? 0.f
                           : input[im_idx];
      }
    }
  }
}

void Im2SequenceCompute::PrepareForRun() {}

void Im2SequenceCompute::Run() {
  auto& ctx = this->ctx_->template As<HostContext>();
  auto& param = this->Param<operators::Im2SequenceParam>();
  auto kernels = param.kernels;
  auto strides = param.strides;
  auto paddings = param.paddings;
  const auto* x_data = param.X->data<float>();
  auto input_dims = param.X->dims();
  int im_num = input_dims[0];
  int im_size = param.X->numel() / im_num;
  int out_cols = input_dims[1] * kernels[0] * kernels[1];
  int total_rows = 0;
  std::vector<uint64_t> im_offset;
  im_offset.push_back(total_rows);
  if (param.Y) {
    const auto* y_data = param.Y->data<float>();
    auto out_strides = param.out_strides;
    std::vector<int> im_real_h;
    std::vector<int> im_real_w;
    std::vector<int> out_h_vec;
    std::vector<int> out_w_vec;
    for (int im_id = 0; im_id < im_num; im_id++) {
      int real_h = static_cast<int>(y_data[im_id * 2 + 0]);
      int real_w = static_cast<int>(y_data[im_id * 2 + 1]);
      int tmp_real_h = (real_h + out_strides[0] - 1) / out_strides[0];
      int tmp_real_w = (real_w + out_strides[1] - 1) / out_strides[1];
      im_real_h.push_back(tmp_real_h);
      im_real_w.push_back(tmp_real_w);
      int out_h =
          (tmp_real_h + paddings[0] + paddings[1] - kernels[0]) / strides[0] +
          1;
      int out_w =
          (tmp_real_w + paddings[2] + paddings[3] - kernels[1]) / strides[1] +
          1;
      out_h_vec.push_back(out_h);
      out_w_vec.push_back(out_w);
      total_rows += out_h * out_w;
      im_offset.push_back(total_rows);
    }
    auto out_dims = param.Out->dims();
    out_dims[0] = total_rows;
    param.Out->Resize(out_dims);
    auto* o_data = param.Out->mutable_data<float>();
    memset(o_data, 0, param.Out->numel() * sizeof(float));

    for (int im_id = 0; im_id < im_num; im_id++) {
      im2sequence(x_data + im_id * im_size,
                  input_dims[1],
                  input_dims[2],
                  input_dims[3],
                  param.kernels[0],
                  param.kernels[1],
                  param.paddings[0],
                  param.paddings[1],
                  param.paddings[2],
                  param.paddings[3],
                  param.strides[0],
                  param.strides[1],
                  out_h_vec[im_id],
                  out_w_vec[im_id],
                  o_data + im_offset[im_id] * out_cols,
                  &ctx);
    }
  } else {
    int out_h =
        (input_dims[2] + paddings[0] + paddings[1] - kernels[0]) / strides[0] +
        1;
    int out_w =
        (input_dims[3] + paddings[2] + paddings[3] - kernels[1]) / strides[1] +
        1;
    auto* o_data = param.Out->mutable_data<float>();
    memset(o_data, 0, param.Out->numel() * sizeof(float));

    for (int im_id = 0; im_id < im_num; im_id++) {
      int out_size_per_im = out_h * out_w * out_cols;
      im2sequence(x_data + im_id * im_size,
                  input_dims[1],
                  input_dims[2],
                  input_dims[3],
                  param.kernels[0],
                  param.kernels[1],
                  param.paddings[0],
                  param.paddings[1],
                  param.paddings[2],
                  param.paddings[3],
                  param.strides[0],
                  param.strides[1],
                  out_h,
                  out_w,
                  o_data + im_id * out_size_per_im,
                  &ctx);
      im_offset.push_back(uint64_t((im_id + 1) * out_h * out_w));
    }
    auto lod = param.Out->mutable_lod();
    lod->resize(1);
    (*lod)[0] = im_offset;
  }
}

}  // namespace host
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(im2sequence,
                     kHost,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::host::Im2SequenceCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kHost))})
    .Finalize();
