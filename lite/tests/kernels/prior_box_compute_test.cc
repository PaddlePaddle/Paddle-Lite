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

#include <gtest/gtest.h>
#include "lite/api/paddle_use_kernels.h"
#include "lite/api/paddle_use_ops.h"
#include "lite/core/arena/framework.h"
#include "lite/core/tensor.h"

namespace paddle {
namespace lite {

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

inline void ExpandAspectRatios(const std::vector<float>& input_aspect_ratior,
                               bool flip,
                               std::vector<float>* output_aspect_ratior) {
  constexpr float epsilon = 1e-6;
  output_aspect_ratior->clear();
  output_aspect_ratior->push_back(1.0f);
  for (size_t i = 0; i < input_aspect_ratior.size(); ++i) {
    float ar = input_aspect_ratior[i];
    bool already_exist = false;
    for (size_t j = 0; j < output_aspect_ratior->size(); ++j) {
      if (fabs(ar - output_aspect_ratior->at(j)) < epsilon) {
        already_exist = true;
        break;
      }
    }
    if (!already_exist) {
      output_aspect_ratior->push_back(ar);
      if (flip) {
        output_aspect_ratior->push_back(1.0f / ar);
      }
    }
  }
}

void prior_box_compute_ref(const lite::Tensor* input,
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
                           const std::vector<std::string>& order_) {
  int win1 = input->dims()[3];
  int hin1 = input->dims()[2];
  DDim out_sh({hin1, win1, prior_num_, 4});
  (*boxes)->Resize(out_sh);
  (*variances)->Resize(out_sh);

  float* _cpu_data = (*boxes)->mutable_data<float>();
  float* _variance_data = (*variances)->mutable_data<float>();

  const int width = input->dims()[3];
  const int height = input->dims()[2];
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
  int step_average = static_cast<int>((step_w + step_h) * 0.5);
  int channel_size = height * width * prior_num_ * 4;
  int idx = 0;
  for (int h = 0; h < height; ++h) {
    for (int w = 0; w < width; ++w) {
      float center_x = (w + offset) * step_w;
      float center_y = (h + offset) * step_h;
      float box_width;
      float box_height;
      if (fixed_size_.size() > 0) {
        for (int s = 0; s < fixed_size_.size(); ++s) {
          int fixed_size = fixed_size_[s];
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
            if (density_size_.size() > 0) {
              CHECK_EQ(fixed_size_.size(), density_size_.size())
                  << "fixed_size should be same with denstiy_size";
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
            // rest of priors : will never come here!!!
            for (int r = 0; r < aspect_ratio_.size(); ++r) {
              float ar = aspect_ratio_[r];

              if (fabs(ar - 1.) < 1e-6) {
                // LOG(INFO) << "returning for aspect == 1";
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
          // ymax
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
          memcpy(_cpu_data + idx, min_buf, sizeof(float) * min_idx);
          idx += min_idx;
          memcpy(_cpu_data + idx, com_buf, sizeof(float) * com_idx);
          idx += com_idx;
          memcpy(_cpu_data + idx, max_buf, sizeof(float) * max_idx);
          idx += max_idx;
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

class DensityPriorBoxComputeTester : public arena::TestCase {
 protected:
  // common attributes for this op.
  std::string ins0 = "Input";
  std::string ins1 = "Image";
  std::string outs0 = "Boxes";
  std::string outs1 = "Variances";
  bool is_flip_;
  bool is_clip_;
  std::vector<float> min_size_;
  std::vector<float> fixed_size_;
  std::vector<float> fixed_ratio_;
  std::vector<int> density_size_;
  std::vector<float> max_size_;
  std::vector<float> aspect_ratio_;
  std::vector<float> variance_;
  int img_w_{0};
  int img_h_{0};
  float step_w_{0};
  float step_h_{0};
  float offset_{0.5};
  int prior_num_{0};
  // priortype: prior_min, prior_max, prior_com
  std::vector<std::string> order_;
  DDim feature_dims_;
  DDim data_dims_;

 public:
  DensityPriorBoxComputeTester(const Place& place,
                               const std::string& alias,
                               bool is_flip,
                               bool is_clip,
                               const std::vector<float>& min_size,
                               const std::vector<float>& fixed_size,
                               const std::vector<float>& fixed_ratio,
                               const std::vector<int>& density_size,
                               const std::vector<float>& max_size,
                               const std::vector<float>& aspect_ratio,
                               const std::vector<float>& variance,
                               int img_w,
                               int img_h,
                               float step_w,
                               float step_h,
                               float offset,
                               int prior_num,
                               // priortype: prior_min, prior_max, prior_com
                               const std::vector<std::string>& order,
                               DDim feature_dims,
                               DDim data_dims)
      : TestCase(place, alias),
        is_flip_(is_flip),
        is_clip_(is_clip),
        min_size_(min_size),
        fixed_size_(fixed_size),
        fixed_ratio_(fixed_ratio),
        density_size_(density_size),
        max_size_(max_size),
        aspect_ratio_(aspect_ratio),
        variance_(variance),
        img_w_(img_w),
        img_h_(img_h),
        step_w_(step_w),
        step_h_(step_h),
        offset_(offset),
        prior_num_(prior_num),
        order_(order),
        feature_dims_(feature_dims),
        data_dims_(data_dims) {}

  // todo get vector<Tensor>
  void RunBaseline(Scope* scope) override {
    auto* inputs0 = scope->FindTensor(ins0);
    auto* inputs1 = scope->FindTensor(ins1);
    auto* outputs0 = scope->NewTensor(outs0);
    auto* outputs1 = scope->NewTensor(outs1);

    CHECK(outputs0);
    CHECK(outputs1);
    CHECK(inputs0);
    CHECK(inputs1);

    prior_box_compute_ref(inputs0,
                          inputs1,
                          &outputs0,
                          &outputs1,
                          min_size_,
                          fixed_size_,
                          fixed_ratio_,
                          density_size_,
                          max_size_,
                          aspect_ratio_,
                          variance_,
                          img_w_,
                          img_h_,
                          step_w_,
                          step_h_,
                          offset_,
                          prior_num_,
                          is_flip_,
                          is_clip_,
                          order_);
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType("density_prior_box");
    op_desc->SetInput("Input", {ins0});
    op_desc->SetInput("Image", {ins1});
    op_desc->SetOutput("Boxes", {outs0});
    op_desc->SetOutput("Variances", {outs1});
    op_desc->SetAttr("flip", is_flip_);
    op_desc->SetAttr("clip", is_clip_);
    op_desc->SetAttr("min_sizes", min_size_);
    op_desc->SetAttr("fixed_sizes", fixed_size_);
    op_desc->SetAttr("fixed_ratios", fixed_ratio_);
    op_desc->SetAttr("density_sizes", density_size_);
    op_desc->SetAttr("max_sizes", max_size_);
    op_desc->SetAttr("aspect_ratios", aspect_ratio_);
    op_desc->SetAttr("variances", variance_);
    op_desc->SetAttr("img_w", img_w_);
    op_desc->SetAttr("img_h", img_h_);
    op_desc->SetAttr("step_w", step_w_);
    op_desc->SetAttr("step_h", step_h_);
    op_desc->SetAttr("offset", offset_);
    op_desc->SetAttr("prior_num", prior_num_);
    op_desc->SetAttr("order", order_);
  }

  void PrepareData() override {
    std::vector<float> feature_data(feature_dims_.production());
    std::vector<float> image_data(data_dims_.production());

    for (int i = 0; i < feature_dims_.production(); ++i) {
      feature_data[i] = i * 1.1 / feature_dims_.production();
    }
    for (int i = 0; i < data_dims_.production(); ++i) {
      image_data[i] = i * 1.2 / data_dims_.production();
    }

    SetCommonTensor(ins0, feature_dims_, feature_data.data());
    SetCommonTensor(ins1, data_dims_, image_data.data());
  }
};

class PriorBoxComputeTester : public arena::TestCase {
 protected:
  // common attributes for this op.
  std::string ins0 = "Input";
  std::string ins1 = "Image";
  std::string outs0 = "Boxes";
  std::string outs1 = "Variances";
  bool is_flip_;
  bool is_clip_;
  std::vector<float> min_size_;
  std::vector<float> max_size_;
  std::vector<float> aspect_ratio_;
  std::vector<float> variance_;
  int img_w_{0};
  int img_h_{0};
  float step_w_{0};
  float step_h_{0};
  float offset_{0.5};
  int prior_num_{0};
  // priortype: prior_min, prior_max, prior_com
  std::vector<std::string> order_;
  DDim feature_dims_;
  DDim data_dims_;

 public:
  PriorBoxComputeTester(const Place& place,
                        const std::string& alias,
                        bool is_flip,
                        bool is_clip,
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
                        // priortype: prior_min, prior_max, prior_com
                        const std::vector<std::string>& order,
                        DDim feature_dims,
                        DDim data_dims)
      : TestCase(place, alias),
        is_flip_(is_flip),
        is_clip_(is_clip),
        min_size_(min_size),
        max_size_(max_size),
        aspect_ratio_(aspect_ratio),
        variance_(variance),
        img_w_(img_w),
        img_h_(img_h),
        step_w_(step_w),
        step_h_(step_h),
        offset_(offset),
        prior_num_(prior_num),
        order_(order),
        feature_dims_(feature_dims),
        data_dims_(data_dims) {}

  void RunBaseline(Scope* scope) override {
    auto* inputs0 = scope->FindTensor(ins0);
    auto* inputs1 = scope->FindTensor(ins1);
    auto* outputs0 = scope->NewTensor(outs0);
    auto* outputs1 = scope->NewTensor(outs1);

    CHECK(outputs0);
    CHECK(outputs1);
    CHECK(inputs0);
    CHECK(inputs1);

    prior_box_compute_ref(inputs0,
                          inputs1,
                          &outputs0,
                          &outputs1,
                          min_size_,
                          std::vector<float>(),
                          std::vector<float>(),
                          std::vector<int>(),
                          max_size_,
                          aspect_ratio_,
                          variance_,
                          img_w_,
                          img_h_,
                          step_w_,
                          step_h_,
                          offset_,
                          prior_num_,
                          is_flip_,
                          is_clip_,
                          order_);
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType("prior_box");
    op_desc->SetInput("Input", {ins0});
    op_desc->SetInput("Image", {ins1});
    op_desc->SetOutput("Boxes", {outs0});
    op_desc->SetOutput("Variances", {outs1});

    op_desc->SetAttr("flip", is_flip_);
    op_desc->SetAttr("clip", is_clip_);
    op_desc->SetAttr("min_sizes", min_size_);
    op_desc->SetAttr("max_sizes", max_size_);
    op_desc->SetAttr("aspect_ratios", aspect_ratio_);
    op_desc->SetAttr("variances", variance_);
    op_desc->SetAttr("img_w", img_w_);
    op_desc->SetAttr("img_h", img_h_);
    op_desc->SetAttr("step_w", step_w_);
    op_desc->SetAttr("step_h", step_h_);
    op_desc->SetAttr("offset", offset_);
    op_desc->SetAttr("prior_num", prior_num_);
    op_desc->SetAttr("order", order_);
  }

  void PrepareData() override {
    std::vector<float> feature_data(feature_dims_.production());
    std::vector<float> image_data(data_dims_.production());

    for (int i = 0; i < feature_dims_.production(); ++i) {
      feature_data[i] = i * 1.1 / feature_dims_.production();
    }
    for (int i = 0; i < data_dims_.production(); ++i) {
      image_data[i] = i * 1.2 / data_dims_.production();
    }

    SetCommonTensor(ins0, feature_dims_, feature_data.data());
    SetCommonTensor(ins1, data_dims_, image_data.data());
  }
};

void test_density_prior_box(Place place) {
  std::vector<float> min_size{60.f};
  std::vector<float> max_size;
  std::vector<float> aspect_ratio{2};
  std::vector<float> variance{0.1f, 0.1f, 0.2f, 0.2f};
  std::vector<float> fixed_size{60, 30};
  std::vector<float> fixed_ratio{1., 2.};
  std::vector<int> density_size{1, 3};
  bool flip = true;
  bool clip = false;
  float step_h = 0;
  float step_w = 0;
  int img_w = 0;
  int img_h = 0;
  float offset = 0.5;
  std::vector<std::string> order;
  std::vector<float> aspect_ratios_vec;
  ExpandAspectRatios(aspect_ratio, flip, &aspect_ratios_vec);
  size_t prior_num = aspect_ratios_vec.size() * min_size.size();
  prior_num += max_size.size();

  if (fixed_size.size() > 0) {
    prior_num = fixed_size.size() * fixed_ratio.size();
  }
  if (density_size.size() > 0) {
    for (int i = 0; i < density_size.size(); ++i) {
      if (fixed_ratio.size() > 0) {
        prior_num += (fixed_ratio.size() * ((pow(density_size[i], 2)) - 1));
      } else {
        prior_num +=
            ((fixed_ratio.size() + 1) * ((pow(density_size[i], 2)) - 1));
      }
    }
  }

  int width = 300;
  int height = 300;
  int channel = 3;
  int num = 1;
  int w_fea = 19;
  int h_fea = 19;
  int c_fea = 512;
  std::unique_ptr<arena::TestCase> tester(
      new DensityPriorBoxComputeTester(place,
                                       "def",
                                       flip,
                                       clip,
                                       min_size,
                                       fixed_size,
                                       fixed_ratio,
                                       density_size,
                                       max_size,
                                       aspect_ratio,
                                       variance,
                                       img_w,
                                       img_h,
                                       step_w,
                                       step_h,
                                       offset,
                                       prior_num,
                                       order,
                                       DDim({num, c_fea, h_fea, w_fea}),
                                       DDim({num, channel, height, width})));
  arena::Arena arena(std::move(tester), place, 2e-5);
  arena.TestPrecision();
}

void test_prior_box(Place place) {
  std::vector<float> min_size{60.f};
  std::vector<float> max_size;
  std::vector<float> aspect_ratio{2.};
  std::vector<float> variance{0.1f, 0.1f, 0.2f, 0.2f};
  bool flip = true;
  bool clip = false;
  float step_h = 0;
  float step_w = 0;
  int img_w = 0;
  int img_h = 0;
  float offset = 0.5;
  std::vector<std::string> order;
  std::vector<float> aspect_ratios_vec;
  ExpandAspectRatios(aspect_ratio, flip, &aspect_ratios_vec);
  size_t prior_num = aspect_ratios_vec.size() * min_size.size();
  prior_num += max_size.size();

  int width = 300;
  int height = 300;
  int channel = 3;
  int num = 1;
  int w_fea = 19;
  int h_fea = 19;
  int c_fea = 128;
  std::unique_ptr<arena::TestCase> tester(
      new PriorBoxComputeTester(place,
                                "def",
                                flip,
                                clip,
                                min_size,
                                max_size,
                                aspect_ratios_vec,
                                variance,
                                img_w,
                                img_h,
                                step_w,
                                step_h,
                                offset,
                                prior_num,
                                order,
                                DDim({num, c_fea, h_fea, w_fea}),
                                DDim({num, channel, height, width})));
  arena::Arena arena(std::move(tester), place, 2e-5);
  arena.TestPrecision();
}

TEST(PriorBox, precision) {
#ifdef LITE_WITH_X86
  Place place(TARGET(kX86));
#endif
#ifdef LITE_WITH_ARM
  Place place(TARGET(kARM));
  test_prior_box(place);
#endif
}

TEST(DensityPriorBox, precision) {
#ifdef LITE_WITH_X86
  Place place(TARGET(kX86));
#endif
#ifdef LITE_WITH_ARM
  Place place(TARGET(kARM));
  test_density_prior_box(place);
#endif
}

}  // namespace lite
}  // namespace paddle
