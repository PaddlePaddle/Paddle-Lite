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

#include "fpga/V1/api.h"
#include "fpga/V1/image.h"
namespace paddle_mobile {
namespace operators {

template <>
bool PSRoiPoolKernel<FPGA, float>::Init(PSRoiPoolParam<FPGA>* param) {
  auto dims = param->input_x_->dims();
  PADDLE_MOBILE_ENFORCE(dims[1] * dims[3] % IMAGE_ALIGNMENT == 0,
                        "data not aligned");

  param->float_input = std::make_shared<Tensor>();
  param->float_input->mutable_data<float>(param->input_x_->dims());
  // param->float_output = std::make_shared<Tensor>();

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

  auto* rois = param->input_rois_;
  int rois_num = rois->dims()[0];
  framework::DDim dims_out_new = framework::make_ddim(
      {rois_num, param->output_->dims()[1], param->output_->dims()[2],
       param->output_->dims()[3]});
  param->output_->Resize(dims_out_new);
  // fpga::format_fp16_ofm(param->output_);

  param->output_->mutable_data<float>(dims_out_new);
  //  auto output = param->float_output.get();
  // param->output_ = output;
  /* args.input_data_type = fpga::DATA_TYPE_FP32;
   args.output_data_type = fpga::DATA_TYPE_FP16;
   args.image.address = output->data<float>();
   args.image.height = (uint32_t)output->dims()[2];
   args.image.width = (uint32_t)output->dims()[3];
   args.image.channels = (uint32_t)output->dims()[1]  ;
   args.output.address = param->output_->mutable_data<half>();
   args.output.scale_address = param->output_->scale;
   param->output_arg = args;*/

  return true;
}

template <typename Dtype>
void PSROIPooling(const Dtype* bottom_data, const Dtype spatial_scale,
                  const int channels, const int height, const int width,
                  const int pooled_height, const int pooled_width,
                  const Dtype* bottom_rois, const int output_dim,
                  const int group_size, Dtype* top_data,
                  // int* mapping_channel,
                  int index, int* rois_batch_id) {
  // The output is in order (n, ctop, ph, pw)
  // static int cnt = 0;
  int pw = index % pooled_width;
  int ph = (index / pooled_width) % pooled_height;
  int ctop = (index / pooled_width / pooled_height) % output_dim;
  int n = index / pooled_width / pooled_height / output_dim;

  // [start, end) interval for spatial sampling
  bottom_rois += n * 4;
  int roi_batch_ind = rois_batch_id[n];  // bottom_rois[0];
  Dtype roi_start_w = static_cast<Dtype>(round(bottom_rois[0])) * spatial_scale;
  Dtype roi_start_h = static_cast<Dtype>(round(bottom_rois[1])) * spatial_scale;
  Dtype roi_end_w =
      static_cast<Dtype>(round(bottom_rois[2]) + 1.) * spatial_scale;
  Dtype roi_end_h =
      static_cast<Dtype>(round(bottom_rois[3]) + 1.) * spatial_scale;

  // Force too small ROIs to be 1x1
  Dtype roi_width = std::max(roi_end_w - roi_start_w, 0.1f);  // avoid 0
  Dtype roi_height = std::max(roi_end_h - roi_start_h, 0.1f);

  // Compute w and h at bottom
  Dtype bin_size_h = roi_height / static_cast<Dtype>(pooled_height);
  Dtype bin_size_w = roi_width / static_cast<Dtype>(pooled_width);

  int hstart = floor(static_cast<Dtype>(ph) * bin_size_h + roi_start_h);
  int wstart = floor(static_cast<Dtype>(pw) * bin_size_w + roi_start_w);
  int hend = ceil(static_cast<Dtype>(ph + 1) * bin_size_h + roi_start_h);
  int wend = ceil(static_cast<Dtype>(pw + 1) * bin_size_w + roi_start_w);
  // Add roi offsets and clip to input boundaries
  hstart = std::min(std::max(hstart, 0), height);
  hend = std::min(std::max(hend, 0), height);
  wstart = std::min(std::max(wstart, 0), width);
  wend = std::min(std::max(wend, 0), width);
  bool is_empty = (hend <= hstart) || (wend <= wstart);

  int gw = pw;
  int gh = ph;
  int c = (ctop * group_size + gh) * group_size + gw;

  bottom_data += (roi_batch_ind * channels + c) * height * width;
  Dtype out_sum = 0;
  for (int h = hstart; h < hend; ++h) {
    for (int w = wstart; w < wend; ++w) {
      int bottom_index = h * width + w;
      out_sum += bottom_data[bottom_index];
    }
  }

  Dtype bin_area = (hend - hstart) * (wend - wstart);
  top_data[index] = is_empty ? 0. : out_sum / bin_area;
}
template <>
void PSRoiPoolKernel<FPGA, float>::Compute(const PSRoiPoolParam<FPGA>& param) {
  auto input_tensor = param.float_input.get();
  fpga::PerformBypass(param.input_arg);
  fpga::fpga_invalidate(input_tensor->data<float>(),
                        input_tensor->numel() * sizeof(float));

  auto* in = input_tensor;
  auto* rois = param.input_rois_;
  auto* out = param.output_;  // param.float_output.get();

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

  auto data_nhwc = in->mutable_data<float>();
  fpga::image::convert_to_chw(&data_nhwc, input_channels, height, width);
  framework::DDim dims_out_new = framework::make_ddim(
      {rois_num, (param.output_)->dims()[1], (((param.output_)->dims()[2])),
       (param.output_)->dims()[3]});
  (param.output_)->Resize(dims_out_new);

  const float* input_data = data_nhwc;  // in->data<float>();
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

  // calculate batch id index for each roi according to LoD
  for (int n = 0; n < rois_batch_size; ++n) {
    for (size_t i = rois_lod[n]; i < rois_lod[n + 1]; ++i) {
      rois_batch_id_data[i] = n;
    }
  }
  auto output_data = out->mutable_data<float>();
  auto input_rois = rois->data<float>();

  // calculate psroipooling, parallel processing can be implemented per ROI

  int index = pooled_height * pooled_width * output_channels * rois_num;
  for (int idx = 0; idx < index; idx++) {
    PSROIPooling<float>(input_data, spatial_scale, input_channels, height,
                        width, pooled_height, pooled_width, input_rois,
                        output_channels, pooled_height, output_data, idx,
                        rois_batch_id_data);
  }
  //
  fpga::image::convert_to_hwc(&output_data, output_channels, pooled_height,
                              pooled_width, rois_num);
  out->reset_data_ptr(output_data);
}

}  // namespace operators
}  // namespace paddle_mobile

#endif  // PSROI_POOL_OP
