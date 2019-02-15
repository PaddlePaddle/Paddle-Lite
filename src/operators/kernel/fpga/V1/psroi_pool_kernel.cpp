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

namespace paddle_mobile {
namespace operators {

template <>
bool PSRoiPoolKernel<FPGA, float>::Init(PSRoiPoolParam<FPGA>* param) {
  auto dims = param->input_x_->dims();
  PADDLE_MOBILE_ENFORCE(dims[1] * dims[3] % IMAGE_ALIGNMENT == 0,
                        "data not aligned");

  param->float_input = std::make_shared<Tensor>();
  param->float_input->mutable_data<float>(param->input_x_->dims());
  param->float_output = std::make_shared<Tensor>();
  param->float_output->mutable_data<float>(param->output_->dims());

  auto input = param->input_x_;
  fpga::BypassArgs args = {fpga::DATA_TYPE_FP16};
  args.input_layout_type = fpga::LAYOUT_HWC;
  args.output_layout_type = fpga::LAYOUT_HWC;
  args.input_data_type = fpga::DATA_TYPE_FP16;
  args.output_data_type = fpga::DATA_TYPE_FP32;
  args.image.address = input->data<half>();
  args.image.height = (uint32_t)input->dims()[2];
  args.image.width = (uint32_t)input->dims()[3];
  args.image.channels = (uint32_t)input->dims()[1];
  args.output.address = param->float_input->mutable_data<float>();
  args.output.scale_address = param->float_input->scale;
  param->input_arg = args;

  fpga::format_fp16_ofm(param->output_);

  input = param->float_output.get();
  args.input_data_type = fpga::DATA_TYPE_FP32;
  args.output_data_type = fpga::DATA_TYPE_FP16;
  args.image.address = input->data<float>();
  args.image.height = (uint32_t)input->dims()[2];
  args.image.width = (uint32_t)input->dims()[3];
  args.image.channels = (uint32_t)input->dims()[1];
  args.output.address = param->output_->mutable_data<half>();
  args.output.scale_address = param->output_->scale;
  param->input_arg = args;

  return true;
}

template <>
void PSRoiPoolKernel<FPGA, float>::Compute(const PSRoiPoolParam<FPGA>& param) {
  auto input_tensor = param.float_input.get();
  fpga::PerformBypass(param.input_arg);
  fpga::fpga_invalidate(input_tensor->data<float>(),
                        input_tensor->numel() * sizeof(float));

  auto* in = input_tensor;
  auto* rois = param.input_rois_;
  auto* out = param.float_output.get();

  auto pooled_height = param.pooled_height_;
  auto pooled_width = param.pooled_width_;
  auto spatial_scale = param.spatial_scale_;
  auto output_channels = param.output_channels_;

  auto in_dims = in->dims();
  int batch_size = in_dims[0];
  int input_channels = in_dims[1];
  int height = in_dims[2];
  int width = in_dims[3];
  int rois_num = rois->dims()[0];

  // TODO   auto in_stride = framework::stride(in_dims);
  // TODO   auto out_stride = framework::stride(out->dims());
  auto in_stride =
      framework::stride({batch_size, height, width, input_channels});
  auto out_stride = framework::stride(
      {out->dims()[0], out->dims()[2], out->dims()[3], out->dims()[1]});

  const float* input_data = in->data<float>();
  framework::Tensor rois_batch_id_list;
  rois_batch_id_list.Resize({rois_num});
  auto rois_batch_id_data = rois_batch_id_list.mutable_data<int>();
  return;

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

  // calculate batch id index for each roi according to LoD
  for (int n = 0; n < rois_batch_size; ++n) {
    for (size_t i = rois_lod[n]; i < rois_lod[n + 1]; ++i) {
      rois_batch_id_data[i] = n;
    }
  }
  auto output_data = out->mutable_data<float>();
  auto input_rois = rois->data<float>();

  // calculate psroipooling, parallel processing can be implemented per ROI
  for (int n = 0; n < rois_num; ++n) {
    // set roi batch id
    int roi_batch_id = rois_batch_id_data[n];

    // [start, end) interval for spatial sampling
    auto offset_input_rois = input_rois + n * 4;
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
    DLOG << 3;

    // calculate each pixel of the output feature map.
    int out_roi_offset = n * out_stride[0];
    for (int c = 0; c < output_channels; ++c) {
      // per category
      // int out_plane_offset = out_roi_offset + c * out_stride[1];
      int out_plane_offset = out_roi_offset + c;
      for (int ph = 0; ph < pooled_height; ++ph) {
        // TODO         int out_row_offset = out_plane_offset + ph *
        // out_stride[2];
        int out_row_offset = out_plane_offset + ph * out_stride[1];
        for (int pw = 0; pw < pooled_width; ++pw) {
          // calculate w and h at input feature map
          int hstart = floor(static_cast<float>(ph) * bin_size_h + roi_start_h);
          int wstart = floor(static_cast<float>(pw) * bin_size_w + roi_start_w);
          int hend =
              ceil(static_cast<float>(ph + 1) * bin_size_h + roi_start_h);
          int wend =
              ceil(static_cast<float>(pw + 1) * bin_size_w + roi_start_w);
          //  Add roi offsets and clip to input boundaries
          hstart = std::min(std::max(hstart, 0), height);
          wstart = std::min(std::max(wstart, 0), width);
          hend = std::min(std::max(hend, 0), height);
          wend = std::min(std::max(wend, 0), width);

          // TODO           int output_index = out_row_offset + pw;
          int output_index = out_row_offset + pw * output_channels;
          int input_channel = (c * pooled_height + ph) * pooled_width + pw;
          // TODO          int input_plane_offset =
          // TODO           roi_batch_id * in_stride[0] + input_channel *
          // in_stride[1];
          int input_plane_offset = roi_batch_id * in_stride[0] + input_channel;
          auto offset_input_data = input_data + input_plane_offset;
          float out_sum = 0.;
          bool is_empty = (hend <= hstart) || (wend <= wstart);
          for (int ih = hstart; ih < hend; ++ih) {
            for (int iw = wstart; iw < wend; ++iw) {
              int input_index = ih * in_stride[1] + iw * input_channel;
              out_sum += offset_input_data[input_index];
            }
          }
          float bin_area = (hend - hstart) * (wend - wstart);
          output_data[output_index] = is_empty ? 0. : out_sum / bin_area;
        }
      }
    }
  }
  fpga::format_image(out);
  fpga::PerformBypass(param.output_arg);
}

}  // namespace operators
}  // namespace paddle_mobile

#endif  // PSROI_POOL_OP
