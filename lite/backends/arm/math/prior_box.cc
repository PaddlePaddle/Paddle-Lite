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

#include "lite/backends/arm/math/prior_box.h"
#include <algorithm>
#include "lite/backends/arm/math/funcs.h"

namespace paddle {
namespace lite {
namespace arm {
namespace math {

const int MALLOC_ALIGN = 64;

void* fast_malloc(size_t size) {
  size_t offset = sizeof(void*) + MALLOC_ALIGN - 1;
  char* p = static_cast<char*>(malloc(offset + size));

  if (!p) {
    return nullptr;
  }

  void* r = reinterpret_cast<void*>(reinterpret_cast<size_t>(p + offset) &
                                    (~(MALLOC_ALIGN - 1)));
  static_cast<void**>(r)[-1] = p;
  memset(r, 0, size);
  return r;
}

void fast_free(void* ptr) {
  if (ptr) {
    free(static_cast<void**>(ptr)[-1]);
  }
}

void density_prior_box(const lite::Tensor* input,
                       const lite::Tensor* image,
                       lite::Tensor** boxes,
                       lite::Tensor** variances,
                       const std::vector<float>& min_size_,
                       const std::vector<float>& fixed_size_,
                       const std::vector<float>& fixed_ratio_,
                       const std::vector<int>& density_size_,
                       const std::vector<float>& max_size_,
                       const std::vector<float>& aspect_ratio_,
                       const std::vector<float>& variance_,
                       int img_w_,
                       int img_h_,
                       float step_w_,
                       float step_h_,
                       float offset_,
                       int prior_num_,
                       bool is_flip_,
                       bool is_clip_,
                       const std::vector<std::string>& order_,
                       bool min_max_aspect_ratios_order) {
  // compute output shape
  int win1 = input->dims()[3];
  int hin1 = input->dims()[2];
  DDim shape_out({hin1, win1, prior_num_, 4});
  (*boxes)->Resize(shape_out);
  (*variances)->Resize(shape_out);

  float* _cpu_data = (*boxes)->mutable_data<float>();
  float* _variance_data = (*variances)->mutable_data<float>();

  const int width = win1;
  const int height = hin1;
  int img_width = img_w_;
  int img_height = img_h_;
  if (img_width == 0 || img_height == 0) {
    img_width = image->dims()[3];
    img_height = image->dims()[2];
  }
  float step_w = step_w_;
  float step_h = step_h_;
  if (step_w == 0 || step_h == 0) {
    step_w = static_cast<float>(img_width) / width;
    step_h = static_cast<float>(img_height) / height;
  }
  float offset = offset_;
  int step_average = static_cast<int>((step_w + step_h) * 0.5);  // add
  int channel_size = height * width * prior_num_ * 4;
  int idx = 0;
  for (int h = 0; h < height; ++h) {
    for (int w = 0; w < width; ++w) {
      float center_x = (w + offset) * step_w;
      float center_y = (h + offset) * step_h;
      float box_width;
      float box_height;
      if (fixed_size_.size() > 0) {
        // add
        for (int s = 0; s < fixed_size_.size(); ++s) {
          int fixed_size = fixed_size_[s];
          int com_idx = 0;
          box_width = fixed_size;
          box_height = fixed_size;

          if (fixed_ratio_.size() > 0) {
            for (int r = 0; r < fixed_ratio_.size(); ++r) {
              float ar = fixed_ratio_[r];
              int density = density_size_[s];
              int shift = step_average / density;
              float box_width_ratio = fixed_size_[s] * sqrt(ar);
              float box_height_ratio = fixed_size_[s] / sqrt(ar);

              for (int p = 0; p < density; ++p) {
                for (int c = 0; c < density; ++c) {
                  float center_x_temp =
                      center_x - step_average / 2.0f + shift / 2.f + c * shift;
                  float center_y_temp =
                      center_y - step_average / 2.0f + shift / 2.f + p * shift;
                  // xmin
                  _cpu_data[idx++] =
                      (center_x_temp - box_width_ratio / 2.f) / img_width >= 0
                          ? (center_x_temp - box_width_ratio / 2.f) / img_width
                          : 0;
                  // ymin
                  _cpu_data[idx++] =
                      (center_y_temp - box_height_ratio / 2.f) / img_height >= 0
                          ? (center_y_temp - box_height_ratio / 2.f) /
                                img_height
                          : 0;
                  // xmax
                  _cpu_data[idx++] =
                      (center_x_temp + box_width_ratio / 2.f) / img_width <= 1
                          ? (center_x_temp + box_width_ratio / 2.f) / img_width
                          : 1;
                  // ymax
                  _cpu_data[idx++] =
                      (center_y_temp + box_height_ratio / 2.f) / img_height <= 1
                          ? (center_y_temp + box_height_ratio / 2.f) /
                                img_height
                          : 1;
                }
              }
            }
          } else {
            // this code for density anchor box
            if (density_size_.size() > 0) {
              CHECK_EQ(fixed_size_.size(), density_size_.size())
                  << "fixed_size_ should be same with density_size_";
              int density = density_size_[s];
              int shift = fixed_size_[s] / density;

              for (int r = 0; r < density; ++r) {
                for (int c = 0; c < density; ++c) {
                  float center_x_temp =
                      center_x - fixed_size / 2.f + shift / 2.f + c * shift;
                  float center_y_temp =
                      center_y - fixed_size / 2.f + shift / 2.f + r * shift;
                  // xmin
                  _cpu_data[idx++] =
                      (center_x_temp - box_width / 2.f) / img_width >= 0
                          ? (center_x_temp - box_width / 2.f) / img_width
                          : 0;
                  // ymin
                  _cpu_data[idx++] =
                      (center_y_temp - box_height / 2.f) / img_height >= 0
                          ? (center_y_temp - box_height / 2.f) / img_height
                          : 0;
                  // xmax
                  _cpu_data[idx++] =
                      (center_x_temp + box_width / 2.f) / img_width <= 1
                          ? (center_x_temp + box_width / 2.f) / img_width
                          : 1;
                  // ymax
                  _cpu_data[idx++] =
                      (center_y_temp + box_height / 2.f) / img_height <= 1
                          ? (center_y_temp + box_height / 2.f) / img_height
                          : 1;
                }
              }
            }

            // rest of priors: will never come here!!!
            for (int r = 0; r < aspect_ratio_.size(); ++r) {
              float ar = aspect_ratio_[r];

              if (fabs(ar - 1.) < 1e-6) {
                continue;
              }

              int density = density_size_[s];
              int shift = fixed_size_[s] / density;
              float box_width_ratio = fixed_size_[s] * sqrt(ar);
              float box_height_ratio = fixed_size_[s] / sqrt(ar);

              for (int p = 0; p < density; ++p) {
                for (int c = 0; c < density; ++c) {
                  float center_x_temp =
                      center_x - fixed_size / 2.f + shift / 2.f + c * shift;
                  float center_y_temp =
                      center_y - fixed_size / 2.f + shift / 2.f + p * shift;
                  // xmin
                  _cpu_data[idx++] =
                      (center_x_temp - box_width_ratio / 2.f) / img_width >= 0
                          ? (center_x_temp - box_width_ratio / 2.f) / img_width
                          : 0;
                  // ymin
                  _cpu_data[idx++] =
                      (center_y_temp - box_height_ratio / 2.f) / img_height >= 0
                          ? (center_y_temp - box_height_ratio / 2.f) /
                                img_height
                          : 0;
                  // xmax
                  _cpu_data[idx++] =
                      (center_x_temp + box_width_ratio / 2.f) / img_width <= 1
                          ? (center_x_temp + box_width_ratio / 2.f) / img_width
                          : 1;
                  // ymax
                  _cpu_data[idx++] =
                      (center_y_temp + box_height_ratio / 2.f) / img_height <= 1
                          ? (center_y_temp + box_height_ratio / 2.f) /
                                img_height
                          : 1;
                }
              }
            }
          }
        }
      } else {
        float* min_buf =
            reinterpret_cast<float*>(fast_malloc(sizeof(float) * 4));
        float* max_buf =
            reinterpret_cast<float*>(fast_malloc(sizeof(float) * 4));
        float* com_buf = reinterpret_cast<float*>(
            fast_malloc(sizeof(float) * aspect_ratio_.size() * 4));

        for (int s = 0; s < min_size_.size(); ++s) {
          int min_idx = 0;
          int max_idx = 0;
          int com_idx = 0;
          int min_size = min_size_[s];
          // first prior: aspect_ratio = 1, size = min_size
          box_width = box_height = min_size;
          //! xmin
          min_buf[min_idx++] = (center_x - box_width / 2.f) / img_width;
          //! ymin
          min_buf[min_idx++] = (center_y - box_height / 2.f) / img_height;
          //! xmax
          min_buf[min_idx++] = (center_x + box_width / 2.f) / img_width;
          //! ymax
          min_buf[min_idx++] = (center_y + box_height / 2.f) / img_height;

          if (max_size_.size() > 0) {
            int max_size = max_size_[s];
            //! second prior: aspect_ratio = 1, size = sqrt(min_size * max_size)
            box_width = box_height = sqrtf(min_size * max_size);
            //! xmin
            max_buf[max_idx++] = (center_x - box_width / 2.f) / img_width;
            //! ymin
            max_buf[max_idx++] = (center_y - box_height / 2.f) / img_height;
            //! xmax
            max_buf[max_idx++] = (center_x + box_width / 2.f) / img_width;
            //! ymax
            max_buf[max_idx++] = (center_y + box_height / 2.f) / img_height;
          }

          //! rest of priors
          for (int r = 0; r < aspect_ratio_.size(); ++r) {
            float ar = aspect_ratio_[r];
            if (fabs(ar - 1.) < 1e-6) {
              continue;
            }
            box_width = min_size * sqrt(ar);
            box_height = min_size / sqrt(ar);
            //! xmin
            com_buf[com_idx++] = (center_x - box_width / 2.f) / img_width;
            //! ymin
            com_buf[com_idx++] = (center_y - box_height / 2.f) / img_height;
            //! xmax
            com_buf[com_idx++] = (center_x + box_width / 2.f) / img_width;
            //! ymax
            com_buf[com_idx++] = (center_y + box_height / 2.f) / img_height;
          }
          if (min_max_aspect_ratios_order) {
            memcpy(_cpu_data + idx, min_buf, sizeof(float) * min_idx);
            idx += min_idx;
            memcpy(_cpu_data + idx, max_buf, sizeof(float) * max_idx);
            idx += max_idx;
            memcpy(_cpu_data + idx, com_buf, sizeof(float) * com_idx);
            idx += com_idx;
          } else {
            memcpy(_cpu_data + idx, min_buf, sizeof(float) * min_idx);
            idx += min_idx;
            memcpy(_cpu_data + idx, com_buf, sizeof(float) * com_idx);
            idx += com_idx;
            memcpy(_cpu_data + idx, max_buf, sizeof(float) * max_idx);
            idx += max_idx;
          }
        }
        fast_free(min_buf);
        fast_free(max_buf);
        fast_free(com_buf);
      }
    }
  }
  //! clip the prior's coordinate such that it is within [0, 1]
  if (is_clip_) {
    for (int d = 0; d < channel_size; ++d) {
      _cpu_data[d] = std::min(std::max(_cpu_data[d], 0.f), 1.f);
    }
  }
  //! set the variance.
  int count = 0;
  for (int h = 0; h < height; ++h) {
    for (int w = 0; w < width; ++w) {
      for (int i = 0; i < prior_num_; ++i) {
        for (int j = 0; j < 4; ++j) {
          _variance_data[count] = variance_[j];
          ++count;
        }
      }
    }
  }
}

void prior_box(const lite::Tensor* input,
               const lite::Tensor* image,
               lite::Tensor** boxes,
               lite::Tensor** variances,
               const std::vector<float>& min_size,
               const std::vector<float>& max_size,
               const std::vector<float>& aspect_ratio,
               const std::vector<float>& variance,
               int img_w,
               int img_h,
               float step_w,
               float step_h,
               float offset,
               int prior_num,
               bool is_flip,
               bool is_clip,
               const std::vector<std::string>& order,
               bool min_max_aspect_ratios_order) {
  density_prior_box(input,
                    image,
                    boxes,
                    variances,
                    min_size,
                    std::vector<float>(),
                    std::vector<float>(),
                    std::vector<int>(),
                    max_size,
                    aspect_ratio,
                    variance,
                    img_w,
                    img_h,
                    step_w,
                    step_h,
                    offset,
                    prior_num,
                    is_flip,
                    is_clip,
                    order,
                    min_max_aspect_ratios_order);
}

}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
