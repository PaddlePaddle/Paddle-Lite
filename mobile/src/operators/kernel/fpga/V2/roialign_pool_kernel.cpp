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

#ifdef ROIALIGN_POOL_OP

#include <cmath>
#include <vector>
#include "operators/kernel/detection_kernel.h"

#include "fpga/V2/api.h"
#include "fpga/V2/image.h"

namespace paddle_mobile {
namespace operators {

template <>
bool RoiAlignPoolKernel<FPGA, float>::Init(RoiAlignPoolParam<FPGA>* param) {
  auto dims = param->input_x_->dims();
  PADDLE_MOBILE_ENFORCE(dims[1] * dims[3] % IMAGE_ALIGNMENT == 0,
                        "data not aligned");

  param->float_input = std::make_shared<Tensor>();
  param->float_input->mutable_data<float>(param->input_x_->dims());

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

  param->output_->mutable_data<float>(dims_out_new);

  return true;
}

template <typename T>
struct PreCalc {
  int pos1;
  int pos2;
  int pos3;
  int pos4;
  T w1;
  T w2;
  T w3;
  T w4;
};

template <typename T>
void pre_calc_for_bilinear_interpolate(
    const int height, const int width, const int pooled_height,
    const int pooled_width, const int iy_upper, const int ix_upper,
    T roi_start_h, T roi_start_w, T bin_size_h, T bin_size_w,
    int roi_bin_grid_h, int roi_bin_grid_w,
    std::vector<PreCalc<T>>& pre_calc) {  // NOLINT
  int pre_calc_index = 0;
  for (int ph = 0; ph < pooled_height; ph++) {
    for (int pw = 0; pw < pooled_width; pw++) {
      for (int iy = 0; iy < iy_upper; iy++) {
        const T yy = roi_start_h + ph * bin_size_h +
                     static_cast<T>(iy + .5f) * bin_size_h /
                         static_cast<T>(roi_bin_grid_h);  // e.g., 0.5, 1.5
        for (int ix = 0; ix < ix_upper; ix++) {
          const T xx = roi_start_w + pw * bin_size_w +
                       static_cast<T>(ix + .5f) * bin_size_w /
                           static_cast<T>(roi_bin_grid_w);

          T x = xx;
          T y = yy;
          // deal with: inverse elements are out of feature map boundary
          if (y < -1.0 || y > height || x < -1.0 || x > width) {
            // empty
            PreCalc<T> pc;
            pc.pos1 = 0;
            pc.pos2 = 0;
            pc.pos3 = 0;
            pc.pos4 = 0;
            pc.w1 = 0;
            pc.w2 = 0;
            pc.w3 = 0;
            pc.w4 = 0;
            pre_calc[pre_calc_index] = pc;
            pre_calc_index += 1;
            continue;
          }

          if (y <= 0) {
            y = 0;
          }
          if (x <= 0) {
            x = 0;
          }

          int y_low = static_cast<int>(y);
          int x_low = static_cast<int>(x);
          int y_high;
          int x_high;

          if (y_low >= height - 1) {
            y_high = y_low = height - 1;
            y = (T)y_low;
          } else {
            y_high = y_low + 1;
          }

          if (x_low >= width - 1) {
            x_high = x_low = width - 1;
            x = (T)x_low;
          } else {
            x_high = x_low + 1;
          }

          T ly = y - y_low;
          T lx = x - x_low;
          T hy = 1. - ly, hx = 1. - lx;
          T w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

          // save weights and indeces
          PreCalc<T> pc;
          pc.pos1 = y_low * width + x_low;
          pc.pos2 = y_low * width + x_high;
          pc.pos3 = y_high * width + x_low;
          pc.pos4 = y_high * width + x_high;
          pc.w1 = w1;
          pc.w2 = w2;
          pc.w3 = w3;
          pc.w4 = w4;
          pre_calc[pre_calc_index] = pc;

          pre_calc_index += 1;
        }
      }
    }
  }
}

template <typename T>
void ROIAlignForward(const int nthreads, const T* bottom_data,
                     const T& spatial_scale, const int channels,
                     const int height, const int width, const int pooled_height,
                     const int pooled_width, const int sampling_ratio,
                     const T* bottom_rois, T* top_data) {
  int n_rois = nthreads / channels / pooled_width / pooled_height;

  for (int n = 0; n < n_rois; n++) {
    int index_n = n * channels * pooled_width * pooled_height;

    // roi could have 4 or 5 columns
    const T* offset_bottom_rois = bottom_rois + n * 4;
    int roi_batch_ind = 0;
    // if (roi_cols == 5) {
    // roi_batch_ind = offset_bottom_rois[0];
    // offset_bottom_rois++;
    // }

    // Do not using rounding; this implementation detail is critical
    T roi_start_w = offset_bottom_rois[0] * spatial_scale;
    T roi_start_h = offset_bottom_rois[1] * spatial_scale;
    T roi_end_w = offset_bottom_rois[2] * spatial_scale;
    T roi_end_h = offset_bottom_rois[3] * spatial_scale;
    // T roi_start_w = round(offset_bottom_rois[0] * spatial_scale);
    // T roi_start_h = round(offset_bottom_rois[1] * spatial_scale);
    // T roi_end_w = round(offset_bottom_rois[2] * spatial_scale);
    // T roi_end_h = round(offset_bottom_rois[3] * spatial_scale);

    // Force malformed ROIs to be 1x1
    T roi_width = std::max(roi_end_w - roi_start_w, (T)1.);
    T roi_height = std::max(roi_end_h - roi_start_h, (T)1.);
    T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
    T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);

    // We use roi_bin_grid to sample the grid and mimic integral
    int roi_bin_grid_h = (sampling_ratio > 0)
                             ? sampling_ratio
                             : ceil(roi_height / pooled_height);  // e.g., = 2
    int roi_bin_grid_w =
        (sampling_ratio > 0) ? sampling_ratio : ceil(roi_width / pooled_width);

    // We do average (integral) pooling inside a bin
    const T count = roi_bin_grid_h * roi_bin_grid_w;  // e.g. = 4

    // we want to precalculate indeces and weights shared by all chanels,
    // this is the key point of optimiation
    std::vector<PreCalc<T>> pre_calc(roi_bin_grid_h * roi_bin_grid_w *
                                     pooled_width * pooled_height);
    pre_calc_for_bilinear_interpolate(
        height, width, pooled_height, pooled_width, roi_bin_grid_h,
        roi_bin_grid_w, roi_start_h, roi_start_w, bin_size_h, bin_size_w,
        roi_bin_grid_h, roi_bin_grid_w, pre_calc);

    for (int c = 0; c < channels; c++) {
      int index_n_c = index_n + c * pooled_width * pooled_height;
      const T* offset_bottom_data =
          bottom_data + (roi_batch_ind * channels + c) * height * width;
      int pre_calc_index = 0;

      for (int ph = 0; ph < pooled_height; ph++) {
        for (int pw = 0; pw < pooled_width; pw++) {
          int index = index_n_c + ph * pooled_width + pw;

          T output_val = 0.;
          for (int iy = 0; iy < roi_bin_grid_h; iy++) {
            for (int ix = 0; ix < roi_bin_grid_w; ix++) {
              PreCalc<T> pc = pre_calc[pre_calc_index];
              output_val += pc.w1 * offset_bottom_data[pc.pos1] +
                            pc.w2 * offset_bottom_data[pc.pos2] +
                            pc.w3 * offset_bottom_data[pc.pos3] +
                            pc.w4 * offset_bottom_data[pc.pos4];

              pre_calc_index += 1;
            }
          }
          output_val /= count;

          top_data[index] = output_val;
        }  // for pw
      }    // for ph
    }      // for c
  }        // for n
}

template <>
void RoiAlignPoolKernel<FPGA, float>::Compute(
    const RoiAlignPoolParam<FPGA>& param) {
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
  auto sampe_ratio = param.sampling_ratio_;

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

  const int index = input_channels * pooled_height * pooled_width * rois_num;
  auto rois_data = rois->data<float>();
  auto top_data = param.output_->mutable_data<float>();
  for (int i = 0; i < index; ++i) {
    ROIAlignForward<float>(index, data_nhwc, spatial_scale, input_channels,
                           height, width, pooled_height, pooled_width,
                           sampe_ratio, rois_data, top_data);
  }

  fpga::image::convert_to_hwc(&top_data, input_channels, pooled_height,
                              pooled_width, rois_num);
  out->reset_data_ptr(top_data);
}

}  // namespace operators
}  // namespace paddle_mobile

#endif  // ROIALIGN_POOL_OP
