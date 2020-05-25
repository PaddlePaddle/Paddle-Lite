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

#include <vector>

#include "lite/api/paddle_place.h"
#include "lite/core/op_registry.h"
#include "lite/core/type_system.h"
#include "lite/kernels/fpga/im2sequence_compute.h"

#include "lite/backends/fpga/KD/float16.hpp"

namespace paddle {
namespace lite {
namespace kernels {
namespace fpga {

using float16 = zynqmp::float16;

void im2sequence(const float16* input,
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
                 float16* out) {
  int window_size = kernel_h * kernel_w;
  int out_rows = out_h * out_w;
  int out_cols = input_c * window_size;
  int H_pad = input_h + pad_top + pad_bottom;
  int W_pad = input_w + pad_left + pad_right;

  float16 zero = zynqmp::float_to_half(0.0f);

  for (int h_id = 0; h_id < out_h; h_id++) {
    for (int w_id = 0; w_id < out_w; w_id++) {
      // consider dilation.
      int start_h = h_id * stride_h - pad_top;
      int start_w = w_id * stride_w - pad_left;
      for (int c_id = 0; c_id < input_c; c_id++) {
        for (int k_h_id = 0; k_h_id < kernel_h; k_h_id++) {
          int in_h_id = start_h + k_h_id;
          bool exceed_flag = (in_h_id < 0) || (in_h_id >= H_pad);
          int out_start_id =
              (h_id * out_w + w_id) * out_cols + c_id * window_size;
          for (int k_w_id = 0; k_w_id < kernel_w; k_w_id++) {
            int in_w_id = start_w + k_w_id;
            exceed_flag = exceed_flag || (in_w_id < 0) || (in_w_id >= W_pad);
            int input_id = (c_id * input_h + in_h_id) * input_w + in_w_id;
            int out_id = out_start_id + k_h_id * kernel_w + k_w_id;
            out[out_id] = exceed_flag ? zero : input[input_id];
          }
        }
      }
    }
  }
}

template <typename T>
void hwc_to_chw(T* chw_data,
                const T* hwc_data,
                int num,
                int channel,
                int height,
                int width) {
  int chw = channel * height * width;
  int wc = width * channel;
  int wh = width * height;
  int index = 0;
  for (int n = 0; n < num; n++) {
    for (int h = 0; h < height; h++) {
      for (int w = 0; w < width; w++) {
        for (int c = 0; c < channel; c++) {
          chw_data[n * chw + c * wh + h * width + w] = hwc_data[index];
          index++;
        }
      }
    }
  }
}

void Im2SequenceCompute::PrepareForRun() {}

void Im2SequenceCompute::Run() {
  auto& param = this->Param<operators::Im2SequenceParam>();
  auto kernels = param.kernels;
  auto strides = param.strides;
  auto paddings = param.paddings;

  const auto* x_data = param.X->data<float16>();
  float16* o_data =
      reinterpret_cast<float16*>(param.Out->mutable_data<float16>());

  float16* o2 = o_data;

  auto input_dims = param.X->dims();
  int im_num = input_dims[0];
  int im_size = param.X->numel() / im_num;

  param.X->ZynqTensor()->syncToCPU();
  float16* chw_data = new float16[param.X->numel()];
  hwc_to_chw<float16>(chw_data,
                      x_data,
                      param.X->dims()[0],
                      param.X->dims()[1],
                      param.X->dims()[2],
                      param.X->dims()[3]);

  const float16* in = chw_data;

  int out_cols = input_dims[1] * kernels[0] * kernels[1];

  int total_rows = 0;
  std::vector<uint64_t> im_offset;
  im_offset.push_back(total_rows);
  if (param.Y) {
    const auto* y_data = param.Y->data<int>();
    auto out_strides = param.out_strides;
    std::vector<int> im_real_h;
    std::vector<int> im_real_w;
    std::vector<int> out_h_vec;
    std::vector<int> out_w_vec;
    for (int im_id = 0; im_id < im_num; im_id++) {
      int real_h = y_data[im_id * 2 + 0];
      int real_w = y_data[im_id * 2 + 1];
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

    int out_offset = 0;
    for (int im_id = 0; im_id < im_num; im_id++) {
      im2sequence(in + im_id * im_size,
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
                  o2 + im_offset[im_id] * out_cols);
    }
  } else {
    int out_h =
        (input_dims[2] + paddings[0] + paddings[1] - kernels[0]) / strides[0] +
        1;
    int out_w =
        (input_dims[3] + paddings[2] + paddings[3] - kernels[1]) / strides[1] +
        1;
    for (int im_id = 0; im_id < im_num; im_id++) {
      int out_size_per_im = out_h * out_w * out_cols;
      im2sequence(in + im_id * im_size,
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
                  o2 + im_id * out_size_per_im);
      im_offset.push_back(uint64_t((im_id + 1) * out_h * out_w));
    }
    auto lod = param.Out->mutable_lod();
    lod->resize(1);
    (*lod)[0] = im_offset;
  }

  delete[] chw_data;
  param.Out->ZynqTensor()->flush();
  param.Out->ZynqTensor()->copyScaleFrom(param.X->ZynqTensor());
}

}  // namespace fpga
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(im2sequence,
                     kFPGA,
                     kFP16,
                     kNHWC,
                     paddle::lite::kernels::fpga::Im2SequenceCompute,
                     def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kFPGA),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kNHWC))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kFPGA),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kNHWC))})
    .Finalize();
