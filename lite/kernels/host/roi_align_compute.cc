// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/kernels/host/roi_align_compute.h"
#include <cmath>
#include <string>
#include <vector>
#include "lite/core/op_registry.h"
#include "lite/core/tensor.h"
#include "lite/core/type_system.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace host {
static constexpr int kROISize = 4;

template <class T>
void PreCalcForBilinearInterpolate(const int height,
                                   const int width,
                                   const int pooled_height,
                                   const int pooled_width,
                                   const int iy_upper,
                                   const int ix_upper,
                                   T roi_ymin,
                                   T roi_xmin,
                                   T bin_size_h,
                                   T bin_size_w,
                                   int roi_bin_grid_h,
                                   int roi_bin_grid_w,
                                   Tensor* pre_pos,
                                   Tensor* pre_w) {
  int pre_calc_index = 0;
  int* pre_pos_data = pre_pos->mutable_data<int>();
  T* pre_w_data = pre_w->mutable_data<T>();
  memset(pre_pos_data, 0, pre_pos->numel() * sizeof(int));
  memset(pre_w_data, 0, pre_w->numel() * sizeof(T));
  for (int ph = 0; ph < pooled_height; ph++) {
    for (int pw = 0; pw < pooled_width; pw++) {
      for (int iy = 0; iy < iy_upper; iy++) {
        // calculate y of sample points
        T y = roi_ymin + ph * bin_size_h +
              static_cast<T>(iy + .5f) * bin_size_h /
                  static_cast<T>(roi_bin_grid_h);
        // calculate x of samle points
        for (int ix = 0; ix < ix_upper; ix++) {
          T x = roi_xmin + pw * bin_size_w +
                static_cast<T>(ix + .5f) * bin_size_w /
                    static_cast<T>(roi_bin_grid_w);
          // deal with elements out of map
          if (y < -1.0 || y > height || x < -1.0 || x > width) {
            for (int i = 0; i < kROISize; ++i) {
              pre_pos_data[i + pre_calc_index * kROISize] = 0;
              pre_w_data[i + pre_calc_index * kROISize] = 0;
            }
            pre_calc_index += 1;
            continue;
          }
          y = y <= 0 ? 0 : y;
          x = x <= 0 ? 0 : x;

          int y_low = static_cast<int>(y);
          int x_low = static_cast<int>(x);
          int y_high;
          int x_high;
          if (y_low >= height - 1) {
            y_high = y_low = height - 1;
            y = static_cast<T>(y_low);
          } else {
            y_high = y_low + 1;
          }
          if (x_low >= width - 1) {
            x_high = x_low = width - 1;
            x = static_cast<T>(x_low);
          } else {
            x_high = x_low + 1;
          }
          T ly = y - y_low, lx = x - x_low;
          T hy = 1. - ly, hx = 1. - lx;
          pre_pos_data[pre_calc_index * kROISize] = y_low * width + x_low;
          pre_pos_data[pre_calc_index * kROISize + 1] = y_low * width + x_high;
          pre_pos_data[pre_calc_index * kROISize + 2] = y_high * width + x_low;
          pre_pos_data[pre_calc_index * kROISize + 3] = y_high * width + x_high;
          pre_w_data[pre_calc_index * kROISize] = hy * hx;
          pre_w_data[pre_calc_index * kROISize + 1] = hy * lx;
          pre_w_data[pre_calc_index * kROISize + 2] = ly * hx;
          pre_w_data[pre_calc_index * kROISize + 3] = ly * lx;
          pre_calc_index += 1;
        }
      }
    }
  }
}

void RoiAlignCompute::Run() {
  auto& param = Param<operators::RoiAlignParam>();
  auto* in = param.X;
  auto* rois = param.ROIs;
  auto* out = param.Out;
  float spatial_scale = param.spatial_scale;
  int pooled_height = param.pooled_height;
  int pooled_width = param.pooled_width;
  int sampling_ratio = param.sampling_ratio;
  bool align = param.align;
  auto in_dims = in->dims();
  int batch_size = in_dims[0];
  int channels = in_dims[1];
  int height = in_dims[2];
  int width = in_dims[3];
  auto rois_dims = rois->dims();
  int rois_num = rois_dims[0];
  auto out_dims = out->dims();
  auto* output_data = out->mutable_data<float>();
  memset(output_data, 0, out->numel() * sizeof(float));

  DDim in_stride({static_cast<int>(in_dims[1] * in_dims[2] * in_dims[3]),
                  static_cast<int>(in_dims[2] * in_dims[3]),
                  static_cast<int>(in_dims[3]),
                  1});
  DDim roi_stride({static_cast<int>(rois_dims[1]), 1});
  DDim out_stride({static_cast<int>(out_dims[1] * out_dims[2] * out_dims[3]),
                   static_cast<int>(out_dims[2] * out_dims[3]),
                   static_cast<int>(out_dims[3]),
                   1});

  int rois_batch_size = 0;
  auto* input_data = in->data<float>();
  auto* rois_num_t = param.RoisNum;
  const int* rois_num_data = nullptr;

  if (param.RoisNum != nullptr) {
    rois_num_data = rois_num_t->data<int>();
    int sum_roi_num = 0;
    for (int i = 0; i < rois_num_t->numel(); i++) {
      sum_roi_num += rois_num_data[i];
    }
    CHECK_EQ(sum_roi_num, rois_num);
  }

  Tensor roi_batch_id_list;
  roi_batch_id_list.Resize({rois_num});
  int* roi_batch_id_data = roi_batch_id_list.mutable_data<int>();
  memset(roi_batch_id_data, 0, roi_batch_id_list.numel() * sizeof(int));

  if (param.RoisNum != nullptr) {
    rois_batch_size = rois_num_t->numel();
    CHECK_EQ(rois_batch_size, batch_size);
    int start = 0;
    for (int n = 0; n < rois_batch_size; ++n) {
      for (int i = start; i < start + rois_num_data[n]; ++i) {
        roi_batch_id_data[i] = n;
      }
      start += rois_num_data[n];
    }
  } else {
    auto lod = rois->lod();
    CHECK_EQ(lod.empty(), false);
    auto rois_lod = rois->lod().back();
    int rois_batch_size = rois_lod.size() - 1;
    CHECK_EQ(rois_batch_size, batch_size);
    int rois_num_with_lod = rois_lod[rois_batch_size];
    CHECK_EQ(rois_num, rois_num_with_lod);
    for (int n = 0; n < rois_batch_size; ++n) {
      for (size_t i = rois_lod[n]; i < rois_lod[n + 1]; ++i) {
        roi_batch_id_data[i] = n;
      }
    }
  }

  auto* rois_data = rois->data<float>();
  float roi_offset = align ? 0.5f : 0.f;
  for (int n = 0; n < rois_num; ++n) {
    int roi_batch_id = roi_batch_id_data[n];
    float roi_xmin = rois_data[0] * spatial_scale - roi_offset;
    float roi_ymin = rois_data[1] * spatial_scale - roi_offset;
    float roi_xmax = rois_data[2] * spatial_scale - roi_offset;
    float roi_ymax = rois_data[3] * spatial_scale - roi_offset;
    float roi_width = roi_xmax - roi_xmin;
    float roi_height = roi_ymax - roi_ymin;
    if (!align) {
      roi_width = std::max(roi_width, 1.f);
      roi_height = std::max(roi_height, 1.f);
    }

    float bin_size_h = roi_height / pooled_height;
    float bin_size_w = roi_width / pooled_width;
    const float* batch_data = input_data + roi_batch_id * in_stride[0];
    int roi_bin_grid_h = (sampling_ratio > 0)
                             ? sampling_ratio
                             : ceil(roi_height / pooled_height);
    int roi_bin_grid_w =
        (sampling_ratio > 0) ? sampling_ratio : ceil(roi_width / pooled_width);
    const int count = std::max(roi_bin_grid_h * roi_bin_grid_w, 1);
    Tensor pre_pos;
    Tensor pre_w;
    int pre_size = count * out_stride[1];
    pre_pos.Resize({pre_size, kROISize});
    pre_w.Resize({pre_size, kROISize});

    PreCalcForBilinearInterpolate<float>(height,
                                         width,
                                         pooled_height,
                                         pooled_width,
                                         roi_bin_grid_h,
                                         roi_bin_grid_w,
                                         roi_ymin,
                                         roi_xmin,
                                         bin_size_h,
                                         bin_size_w,
                                         roi_bin_grid_h,
                                         roi_bin_grid_w,
                                         &pre_pos,
                                         &pre_w);

    const int* pre_pos_data = pre_pos.data<int>();
    const float* pre_w_data = pre_w.data<float>();
    for (int c = 0; c < channels; c++) {
      int pre_calc_index = 0;
      for (int ph = 0; ph < pooled_height; ph++) {
        for (int pw = 0; pw < pooled_width; pw++) {
          const int pool_index = ph * pooled_width + pw;
          float output_val = 0;
          for (int iy = 0; iy < roi_bin_grid_h; iy++) {
            for (int ix = 0; ix < roi_bin_grid_w; ix++) {
              for (int i = 0; i < kROISize; i++) {
                int pos = pre_pos_data[pre_calc_index * kROISize + i];
                float w = pre_w_data[pre_calc_index * kROISize + i];
                output_val += w * batch_data[pos];
              }
              pre_calc_index += 1;
            }
          }
          output_val /= count;
          output_data[pool_index] = output_val;
        }
      }
      batch_data += in_stride[1];
      output_data += out_stride[1];
    }
    rois_data += roi_stride[0];
  }
}

}  // namespace host
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(roi_align,
                     kHost,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::host::RoiAlignCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindInput("ROIs", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindInput("RoisNum",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindPaddleOpVersion("roi_align", 1)
    .Finalize();
