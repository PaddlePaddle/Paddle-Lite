// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/kernels/host/roi_perspective_transform_compute.h"
#include <algorithm>
#include <cmath>

namespace paddle {
namespace lite {
namespace kernels {
namespace host {

template <typename T>
bool GT_E(T a, T b) {
  return (a > b) || fabs(a - b) < 1e-4;
}

template <typename T>
bool LT_E(T a, T b) {
  return (a < b) || fabs(a - b) < 1e-4;
}

template <typename T>
bool GT(T a, T b) {
  return (a - b) > 1e-4;
}

/**
 * Get the matrix of perspective transform.
 *
 * dx1 = x1 - x2
 * dx2 = x3 - x2
 * dx3 = x0 - x1 + x2 - x3
 * dy1 = y1 - y2
 * dy2 = y3 - y2
 * dy3 = y0 - y1 + y2 - y3
 *
 * a11 = (x1 - x0 + a31 * (w - 1) * x1) / (w - 1)
 * a12 = (x3 - x0 + a32 * (h - 1) * x3) / (h - 1)
 * a13 = x0
 * a21 = (y1 - y0 + a31 * (w - 1) * y1) / (w - 1)
 * a22 = (y3 - y0 + a32 * (h - 1) * y3) / (h - 1)
 * a23 = y0
 * a31 = (dx3 * dy2 - dx2 * dy3) / (dx1 * dy2 - dx2 * dy1) / (w - 1)
 * a32 = (dx1 * dy3 - dx3 * dy1) / (dx1 * dy2 - dx2 * dy1) / (h - 1)
 * a33 = 1
 */
template <typename T>
void get_transform_matrix(const int transformed_width,
                          const int transformed_height,
                          T roi_x[],
                          T roi_y[],
                          T matrix[]) {
  T x0 = roi_x[0];
  T x1 = roi_x[1];
  T x2 = roi_x[2];
  T x3 = roi_x[3];
  T y0 = roi_y[0];
  T y1 = roi_y[1];
  T y2 = roi_y[2];
  T y3 = roi_y[3];

  // Estimate the height and width of RoI
  T len1 = sqrt((x0 - x1) * (x0 - x1) + (y0 - y1) * (y0 - y1));
  T len2 = sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));
  T len3 = sqrt((x2 - x3) * (x2 - x3) + (y2 - y3) * (y2 - y3));
  T len4 = sqrt((x3 - x0) * (x3 - x0) + (y3 - y0) * (y3 - y0));
  T estimated_height = (len2 + len4) / 2.0;
  T estimated_width = (len1 + len3) / 2.0;

  // Get the normalized height and normalized width
  int normalized_height = std::max(2, transformed_height);
  int normalized_width =
      std::round(estimated_width * (normalized_height - 1) / estimated_height) +
      1;
  normalized_width = std::max(2, std::min(normalized_width, transformed_width));

  T dx1 = x1 - x2;
  T dx2 = x3 - x2;
  T dx3 = x0 - x1 + x2 - x3;
  T dy1 = y1 - y2;
  T dy2 = y3 - y2;
  T dy3 = y0 - y1 + y2 - y3;

  matrix[6] = (dx3 * dy2 - dx2 * dy3) / (dx1 * dy2 - dx2 * dy1 + 1e-5) /
              (normalized_width - 1);
  matrix[7] = (dx1 * dy3 - dx3 * dy1) / (dx1 * dy2 - dx2 * dy1 + 1e-5) /
              (normalized_height - 1);
  matrix[8] = 1;

  matrix[3] = (y1 - y0 + matrix[6] * (normalized_width - 1) * y1) /
              (normalized_width - 1);
  matrix[4] = (y3 - y0 + matrix[7] * (normalized_height - 1) * y3) /
              (normalized_height - 1);
  matrix[5] = y0;

  matrix[0] = (x1 - x0 + matrix[6] * (normalized_width - 1) * x1) /
              (normalized_width - 1);
  matrix[1] = (x3 - x0 + matrix[7] * (normalized_height - 1) * x3) /
              (normalized_height - 1);
  matrix[2] = x0;
}

/**
 * Get the source coordinates in the input feature map.
 *
 * (u, v, w)^matrix = matrix * (out_w, out_h, 1)^matrix
 *
 * in_w = u / w
 * in_h = v / w
 *
 */
template <typename T>
void get_source_coords(T matrix[], int out_w, int out_h, T* in_w, T* in_h) {
  T u = matrix[0] * out_w + matrix[1] * out_h + matrix[2];
  T v = matrix[3] * out_w + matrix[4] * out_h + matrix[5];
  T w = matrix[6] * out_w + matrix[7] * out_h + matrix[8];

  in_w[0] = u / w;
  in_h[0] = v / w;
}

/*
*check if (x, y) is in the boundary of roi
*/
template <typename T>
bool in_quad(T x, T y, T roi_x[], T roi_y[]) {
  for (int i = 0; i < 4; i++) {
    T xs = roi_x[i];
    T ys = roi_y[i];
    T xe = roi_x[(i + 1) % 4];
    T ye = roi_y[(i + 1) % 4];
    if (fabs(ys - ye) < 1e-4) {
      if (fabs(y - ys) < 1e-4 && fabs(y - ye) < 1e-4 &&
          GT_E<T>(x, std::min(xs, xe)) && LT_E<T>(x, std::max(xs, xe))) {
        return true;
      }
    } else {
      T intersec_x = (y - ys) * (xe - xs) / (ye - ys) + xs;
      if (fabs(intersec_x - x) < 1e-4 && GT_E<T>(y, std::min(ys, ye)) &&
          LT_E<T>(y, std::max(ys, ye))) {
        return true;
      }
    }
  }

  int n_cross = 0;
  for (int i = 0; i < 4; i++) {
    T xs = roi_x[i];
    T ys = roi_y[i];
    T xe = roi_x[(i + 1) % 4];
    T ye = roi_y[(i + 1) % 4];
    if (fabs(ys - ye) < 1e-4) {
      continue;
    }
    if (LT_E<T>(y, std::min(ys, ye)) || GT<T>(y, std::max(ys, ye))) {
      continue;
    }
    T intersec_x = (y - ys) * (xe - xs) / (ye - ys) + xs;
    if (fabs(intersec_x - x) < 1e-4) {
      return true;
    }
    if (GT<T>(intersec_x, x)) {
      n_cross++;
    }
  }
  return (n_cross % 2 == 1);
}

/**
 * Perform bilinear interpolation in the input feature map.
 */
template <typename T>
void bilinear_interpolate(const T* in_data,
                          const int channels,
                          const int width,
                          const int height,
                          int in_n,
                          int in_c,
                          T in_w,
                          T in_h,
                          T* val) {
  // Deal with cases that source coords are out of feature map boundary
  if (GT_E<T>(-0.5, in_w) || GT_E<T>(in_w, width - 0.5) ||
      GT_E<T>(-0.5, in_h) || GT_E<T>(in_h, height - 0.5)) {
    // empty
    val[0] = 0.0;
    return;
  }

  if (GT_E<T>(0, in_w)) {
    in_w = 0;
  }
  if (GT_E<T>(0, in_h)) {
    in_h = 0;
  }

  int in_w_floor = floor(in_w);
  int in_h_floor = floor(in_h);
  int in_w_ceil;
  int in_h_ceil;

  if (GT_E<T>(in_w_floor, width - 1)) {
    in_w_ceil = in_w_floor = width - 1;
    in_w = static_cast<T>(in_w_floor);
  } else {
    in_w_ceil = in_w_floor + 1;
  }

  if (GT_E<T>(in_h_floor, height - 1)) {
    in_h_ceil = in_h_floor = height - 1;
    in_h = static_cast<T>(in_h_floor);
  } else {
    in_h_ceil = in_h_floor + 1;
  }
  T w_floor = in_w - in_w_floor;
  T h_floor = in_h - in_h_floor;
  T w_ceil = 1 - w_floor;
  T h_ceil = 1 - h_floor;
  const T* data = in_data + (in_n * channels + in_c) * height * width;
  // Do bilinear interpolation
  T v1 = data[in_h_floor * width + in_w_floor];
  T v2 = data[in_h_ceil * width + in_w_floor];
  T v3 = data[in_h_ceil * width + in_w_ceil];
  T v4 = data[in_h_floor * width + in_w_ceil];
  T w1 = w_ceil * h_ceil;
  T w2 = w_ceil * h_floor;
  T w3 = w_floor * h_floor;
  T w4 = w_floor * h_ceil;
  val[0] = w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4;
}

template <class T>
void RoiPerspectiveTransformCompute<T>::Run() {
  auto& param = this->template Param<param_t>();
  auto* in = param.x;
  auto* rois = param.rois;
  auto* out = param.out;
  auto* mask = param.mask;
  auto* out_transform_matrix = param.transfor_matrix;
  auto transformed_height = param.transformed_height;
  auto transformed_width = param.transformed_width;
  auto spatial_scale = param.spatial_scale;

  auto in_dims = in->dims();
  int channels = in_dims[1];
  int in_height = in_dims[2];
  int in_width = in_dims[3];
  int rois_num = rois->dims()[0];

  const T* input_data = in->template data<T>();
  int* mask_data = mask->template mutable_data<int>();

  Tensor roi2image;
  roi2image.Resize({static_cast<int64_t>(rois_num)});
  int* roi2image_data = roi2image.template mutable_data<int>();
  auto lod = rois->lod().back();
  for (size_t i = 0; i < lod.size() - 1; ++i) {
    for (size_t j = lod[i]; j < lod[i + 1]; ++j) {
      roi2image_data[j] = i;
    }
  }

  T* output_data = out->template mutable_data<T>();
  const T* rois_data = rois->template data<T>();
  T* transform_matrix = out_transform_matrix->template mutable_data<T>();

  for (int n = 0; n < rois_num; ++n) {
    const T* n_rois = rois_data + n * 8;
    T roi_x[4];
    T roi_y[4];
    for (int k = 0; k < 4; ++k) {
      roi_x[k] = n_rois[2 * k] * spatial_scale;
      roi_y[k] = n_rois[2 * k + 1] * spatial_scale;
    }
    int image_id = roi2image_data[n];
    // Get transform matrix
    T matrix[9];
    get_transform_matrix<T>(
        transformed_width, transformed_height, roi_x, roi_y, matrix);
    for (int i = 0; i < 9; i++) {
      transform_matrix[n * 9 + i] = matrix[i];
    }
    for (int c = 0; c < channels; ++c) {
      for (int out_h = 0; out_h < transformed_height; ++out_h) {
        for (int out_w = 0; out_w < transformed_width; ++out_w) {
          int out_index =
              n * channels * transformed_height * transformed_width +
              c * transformed_height * transformed_width +
              out_h * transformed_width + out_w;
          T in_w, in_h;
          get_source_coords<T>(matrix, out_w, out_h, &in_w, &in_h);
          if (in_quad<T>(in_w, in_h, roi_x, roi_y)) {
            if (GT_E<T>(-0.5, in_w) ||
                GT_E<T>(in_w, static_cast<T>(in_width - 0.5)) ||
                GT_E<T>(-0.5, in_h) ||
                GT_E<T>(in_h, static_cast<T>(in_height - 0.5))) {
              output_data[out_index] = 0.0;
              mask_data[(n * transformed_height + out_h) * transformed_width +
                        out_w] = 0;
            } else {
              bilinear_interpolate(input_data,
                                   channels,
                                   in_width,
                                   in_height,
                                   image_id,
                                   c,
                                   in_w,
                                   in_h,
                                   output_data + out_index);
              mask_data[(n * transformed_height + out_h) * transformed_width +
                        out_w] = 1;
            }
          } else {
            output_data[out_index] = 0.0;
            mask_data[(n * transformed_height + out_h) * transformed_width +
                      out_w] = 0;
          }
        }
      }
    }
  }
}

}  // namespace host
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(
    roi_perspective_transform,
    kHost,
    kFloat,
    kNCHW,
    paddle::lite::kernels::host::RoiPerspectiveTransformCompute<float>,
    def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindInput("ROIs", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("Mask",
                {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindOutput("TransformMatrix", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("Out2InIdx",
                {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindOutput("Out2InWeights", {LiteType::GetTensorTy(TARGET(kHost))})
    .Finalize();
