/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#ifdef PSROI_POOL_OP

#include <cmath>
#include <vector>
#include "operators/kernel/detection_kernel.h"

#include "fpga/V2/api.h"
#include "fpga/V2/image.h"
namespace paddle_mobile {
namespace operators {

template <>
bool PSRoiPoolKernel<FPGA, float>::Init(PSRoiPoolParam<FPGA>* param) {
  auto dims = param->input_x_->dims();
  PADDLE_MOBILE_ENFORCE(dims[1] * dims[3] % IMAGE_ALIGNMENT == 0,
                        "data not aligned");

  param->float_input = std::make_shared<Tensor>();
  param->float_input->mutable_data<float>(param->input_x_->dims());

  auto* rois = param->input_rois_;
  int rois_num = rois->dims()[0];
  framework::DDim dims_out_new = framework::make_ddim(
      {rois_num, param->output_->dims()[1], param->output_->dims()[2],
       param->output_->dims()[3]});
  param->output_->Resize(dims_out_new);

  param->output_->mutable_data<float>(dims_out_new);
  return true;
}

template <typename Dtype>
void PSROIPoolingForward(const int8_t* bottom_data, const int height,
                         const int width, const int input_channel,
                         Dtype* top_data, const int pooled_height,
                         const int pooled_width, const int output_channel,
                         const Dtype* bottom_rois, const Dtype Bin_size_h,
                         const Dtype Bin_size_w, const Dtype roi_start_h,
                         const Dtype roi_start_w, const int pw, const int ph,
                         float scale, const int roi_batch_ind) {
  int hstart = floor(static_cast<Dtype>(ph) * Bin_size_h + roi_start_h);
  int wstart = floor(static_cast<Dtype>(pw) * Bin_size_w + roi_start_w);
  int hend = ceil(static_cast<Dtype>(ph + 1) * Bin_size_h + roi_start_h);
  int wend = ceil(static_cast<Dtype>(pw + 1) * Bin_size_w + roi_start_w);

  // Add roi offsets and clip to input boundaries
  hstart = std::min(std::max(hstart, 0), height);
  hend = std::min(std::max(hend, 0), height);
  wstart = std::min(std::max(wstart, 0), width);
  wend = std::min(std::max(wend, 0), width);
  bool is_empty = (hend <= hstart) || (wend <= wstart);

  float avg_pixels_c[output_channel] = {0};
  int sum_pixels_c[output_channel] = {0};
  int8_t pixels_c[output_channel] = {0};
  if (!is_empty) {
    Dtype bin_area = (hend - hstart) * (wend - wstart);
    float scale_fuse = scale / bin_area;

    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        int pixel_offset = (h * width + w) * input_channel;
        for (int output_c = 0; output_c < output_channel; output_c++) {
          int input_channel_offset = output_c * pooled_height * pooled_width;
          int input_bias =
              pixel_offset + input_channel_offset + ph * pooled_width + pw;
          pixels_c[output_c] = bottom_data[input_bias];
        }

        for (int output_c = 0; output_c < output_channel; output_c++) {
          sum_pixels_c[output_c] += pixels_c[output_c];
        }
      }
    }
    for (int output_c = 0; output_c < output_channel; output_c++) {
      avg_pixels_c[output_c] = sum_pixels_c[output_c] * scale_fuse;
    }
  }

  int output_index_base = (ph * pooled_width + pw) * output_channel;
  top_data += output_index_base;
  memcpy(top_data, avg_pixels_c, output_channel * 4);
}

template <>
void PSRoiPoolKernel<FPGA, float>::Compute(const PSRoiPoolParam<FPGA>& param) {
  auto input_tensor = param.input_x_;
  auto input_data = input_tensor->data<int8_t>();
  auto scale = input_tensor->scale[0] / 127.0;
  fpga::fpga_invalidate(input_data, input_tensor->numel() * sizeof(int8_t));
  auto* rois = param.input_rois_;
  auto* out = param.output_;

  auto pooled_height = param.pooled_height_;
  auto pooled_width = param.pooled_width_;
  auto spatial_scale = param.spatial_scale_;
  auto output_channels = param.output_channels_;

  auto in_dims = input_tensor->dims();
  int batch_size = in_dims[0];
  int input_channels = in_dims[1];
  int height = in_dims[2];
  int width = in_dims[3];
  int rois_num = rois->dims()[0];

  framework::DDim dims_out_new = framework::make_ddim(
      {rois_num, (param.output_)->dims()[1], (((param.output_)->dims()[2])),
       (param.output_)->dims()[3]});

  (param.output_)->Resize(dims_out_new);

  framework::Tensor rois_batch_id_list;
  rois_batch_id_list.Resize({rois_num});
  auto rois_batch_id_data = rois_batch_id_list.mutable_data<int>();

  PADDLE_MOBILE_ENFORCE(rois->NumLevels() > 0, "ROIS should not be empty");

  auto rois_lod = rois->lod().back();
  int rois_batch_size = rois_lod.size() - 1;
  PADDLE_MOBILE_ENFORCE(
      rois_batch_size == batch_size,
      "the rois_batch_size and input(X) batch_size should be the same.");
  int rois_num_with_lod = rois_lod[rois_batch_size];
  PADDLE_MOBILE_ENFORCE(rois_num_with_lod == rois_num,
                        "the rois_num from input and lod must be the same");

  PADDLE_MOBILE_ENFORCE(
      input_channels == output_channels * pooled_height * pooled_width,
      "the channels of input X should equal the product of "
      "output_channels x pooled_height x pooled_width");

  auto output_data = out->mutable_data<float>();
  auto input_rois = rois->data<float>();

  for (int n = 0; n < rois_num; ++n) {
    auto offset_input_rois = input_rois + n * 4;
    auto offset_output_data =
        output_data + pooled_height * pooled_width * output_channels * n;

    auto roi_start_w =
        static_cast<float>(round(offset_input_rois[0])) * spatial_scale;
    auto roi_start_h =
        static_cast<float>(round(offset_input_rois[1])) * spatial_scale;
    auto roi_end_w =
        static_cast<float>(round(offset_input_rois[2]) + 1.) * spatial_scale;
    auto roi_end_h =
        static_cast<float>(round(offset_input_rois[3]) + 1.) * spatial_scale;

    // Force too small rois to be 1 x 1
    auto roi_height = std::max(roi_end_h - roi_start_h, 0.1f);  // avoid 0
    auto roi_width = std::max(roi_end_w - roi_start_w, 0.1f);

    // Compute bin size w and h at input feature map
    auto bin_size_h = roi_height / static_cast<float>(pooled_height);
    auto bin_size_w = roi_width / static_cast<float>(pooled_width);

    int roi_batch_ind = rois_batch_id_data[n];

    for (int ph = 0; ph < pooled_height; ph++) {
      for (int pw = 0; pw < pooled_width; pw++) {
        PSROIPoolingForward<float>(input_data, height, width, input_channels,
                                   offset_output_data, pooled_height,
                                   pooled_width, output_channels, input_rois,
                                   bin_size_h, bin_size_w, roi_start_h,
                                   roi_start_w, pw, ph, scale, roi_batch_ind);
      }
    }
  }
}

}  // namespace operators
}  // namespace paddle_mobile

#endif  // PSROI_POOL_OP
