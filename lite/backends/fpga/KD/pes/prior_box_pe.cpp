/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <math.h>
#include <algorithm>
#include <vector>

#include "lite/backends/fpga/KD/pes/prior_box_pe.hpp"

namespace paddle {
namespace zynqmp {

struct Transform {
  template <typename InputIter, typename OutputIter, typename UnaryOperation>
  void operator()(InputIter first,
                  InputIter last,
                  OutputIter result,
                  UnaryOperation op) {
    std::transform(first, last, result, op);
  }

  template <typename InputIter1,
            typename InputIter2,
            typename OutputIter,
            typename BinaryOperation>
  void operator()(InputIter1 first1,
                  InputIter1 last1,
                  InputIter2 first2,
                  OutputIter result,
                  BinaryOperation op) {
    std::transform(first1, last1, first2, result, op);
  }
};

inline void ExpandAspectRatios(const std::vector<float> &input_aspect_ratior,
                               bool flip,
                               std::vector<float> *output_aspect_ratior) {
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

template <typename T>
struct ClipFunctor {
  inline T operator()(T in) const {
    return std::min<T>(std::max<T>(in, 0.), 1.);
  }
};

void PriorBoxPE::compute_prior_box() {
  PriorBoxParam &param = param_;
  Tensor *input = param.input;
  Shape &input_shape = input->shape();

  Tensor *input_image = param.image;
  Shape &image_shape = input_image->shape();

  const auto &min_sizes = param.minSizes;
  const auto &max_sizes = param.maxSizes;
  const auto &input_aspect_ratio = param.aspectRatios;
  const bool &flip = param.flip;
  const bool &clip = param.clip;
  const float &step_w = param.stepW;
  const float &step_h = param.stepH;
  const float &offset = param.offset;

  Tensor *output_boxes = this->cachedBoxes_.get();
  Tensor *output_variances = this->cachedVariances_.get();

  Tensor boxes;
  Tensor variances;

  float *output_boxes_dataptr =
      boxes.mutableData<float>(FP32, output_boxes->shape());
  memset(output_boxes_dataptr, 0, boxes.memorySize());
  float *output_variances_dataptr =
      variances.mutableData<float>(FP32, output_boxes->shape());

  std::vector<float> aspect_ratios;
  ExpandAspectRatios(input_aspect_ratio, flip, &aspect_ratios);

  auto img_width = image_shape.width();
  auto img_height = image_shape.height();
  auto feature_width = input_shape.width();
  auto feature_height = input_shape.height();

  auto stride0 = output_boxes->shape().channel() *
                 output_boxes->shape().height() * output_boxes->shape().width();
  auto stride1 = output_boxes->shape().height() * output_boxes->shape().width();
  auto stride2 = output_boxes->shape().width();

  float step_width = step_w;
  float step_height = step_h;
  if (step_w == 0 || step_h == 0) {
    step_width = static_cast<float>(img_width) / feature_width;
    step_height = static_cast<float>(img_height) / feature_height;
  }

  int num_priors = aspect_ratios.size() * min_sizes.size();
  if (!max_sizes.empty()) {
    num_priors += max_sizes.size();
  }

  for (int h = 0; h < feature_height; ++h) {
    for (int w = 0; w < feature_width; ++w) {
      /// map origin image
      float center_x = (w + offset) * step_width;
      float center_y = (h + offset) * step_height;
      float box_width, box_height;
      int idx = 0;
      for (size_t s = 0; s < min_sizes.size(); ++s) {
        auto min_size = min_sizes[s];
        if (param.minMaxAspectRatiosOrder) {
          box_width = box_height = min_size / 2.;
          output_boxes_dataptr[h * stride0 + w * stride1 + idx * stride2 + 0] =
              (center_x - box_width) / img_width;
          output_boxes_dataptr[h * stride0 + w * stride1 + idx * stride2 + 1] =
              (center_y - box_height) / img_height;
          output_boxes_dataptr[h * stride0 + w * stride1 + idx * stride2 + 2] =
              (center_x + box_width) / img_width;
          output_boxes_dataptr[h * stride0 + w * stride1 + idx * stride2 + 3] =
              (center_y + box_height) / img_height;
          idx++;

          if (max_sizes.size() > 0) {
            auto max_size = max_sizes[s];
            // square prior with size sqrt(minSize * maxSize)
            box_width = box_height = sqrt(min_size * max_size) / 2.;
            output_boxes_dataptr[h * stride0 + w * stride1 + idx * stride2 +
                                 0] = (center_x - box_width) / img_width;
            output_boxes_dataptr[h * stride0 + w * stride1 + idx * stride2 +
                                 1] = (center_y - box_height) / img_height;
            output_boxes_dataptr[h * stride0 + w * stride1 + idx * stride2 +
                                 2] = (center_x + box_width) / img_width;
            output_boxes_dataptr[h * stride0 + w * stride1 + idx * stride2 +
                                 3] = (center_y + box_height) / img_height;

            idx++;
          }

          // priors with different aspect ratios
          for (float ar : aspect_ratios) {
            if (fabs(ar - 1.) < 1e-6) {
              continue;
            }
            box_width = min_size * sqrt(ar) / 2.;
            box_height = min_size / sqrt(ar) / 2.;
            output_boxes_dataptr[h * stride0 + w * stride1 + idx * stride2 +
                                 0] = (center_x - box_width) / img_width;
            output_boxes_dataptr[h * stride0 + w * stride1 + idx * stride2 +
                                 1] = (center_y - box_height) / img_height;
            output_boxes_dataptr[h * stride0 + w * stride1 + idx * stride2 +
                                 2] = (center_x + box_width) / img_width;
            output_boxes_dataptr[h * stride0 + w * stride1 + idx * stride2 +
                                 3] = (center_y + box_height) / img_height;

            idx++;
          }

        } else {
          // priors with different aspect ratios
          for (float ar : aspect_ratios) {
            box_width = min_size * sqrt(ar) / 2.;
            box_height = min_size / sqrt(ar) / 2.;
            output_boxes_dataptr[h * stride0 + w * stride1 + idx * stride2 +
                                 0] = (center_x - box_width) / img_width;
            output_boxes_dataptr[h * stride0 + w * stride1 + idx * stride2 +
                                 1] = (center_y - box_height) / img_height;
            output_boxes_dataptr[h * stride0 + w * stride1 + idx * stride2 +
                                 2] = (center_x + box_width) / img_width;
            output_boxes_dataptr[h * stride0 + w * stride1 + idx * stride2 +
                                 3] = (center_y + box_height) / img_height;
            idx++;
          }
          if (!max_sizes.empty()) {
            auto max_size = max_sizes[s];
            // square prior with size sqrt(minSize * maxSize)
            box_width = box_height = sqrt(min_size * max_size) / 2.;
            output_boxes_dataptr[h * stride0 + w * stride1 + idx * stride2 +
                                 0] = (center_x - box_width) / img_width;
            output_boxes_dataptr[h * stride0 + w * stride1 + idx * stride2 +
                                 1] = (center_y - box_height) / img_height;
            output_boxes_dataptr[h * stride0 + w * stride1 + idx * stride2 +
                                 2] = (center_x + box_width) / img_width;
            output_boxes_dataptr[h * stride0 + w * stride1 + idx * stride2 +
                                 3] = (center_y + box_height) / img_height;
            idx++;
          }
        }
      }
    }
  }
  if (clip) {
    for (int i = 0; i < output_boxes->shape().numel(); i++) {
      float value = output_boxes_dataptr[i];
      value = std::min(std::max(0.0f, value), 1.0f);
      output_boxes_dataptr[i] = value;
    }
  }

  if ((param.variances.size() != 4)) {
    // TODO(chonwhite) throw error;
  }

  int64_t box_num = feature_height * feature_width * num_priors;

  for (int i = 0; i < box_num; i++) {
    output_variances_dataptr[4 * i] = param.variances[0];
    output_variances_dataptr[4 * i + 1] = param.variances[1];
    output_variances_dataptr[4 * i + 2] = param.variances[2];
    output_variances_dataptr[4 * i + 3] = param.variances[3];
  }

  boxes.flush();
  variances.flush();
  output_boxes->copyFrom(&boxes);
  output_variances->copyFrom(&variances);
}

void PriorBoxPE::apply() {}

bool PriorBoxPE::dispatch() {
  if (cachedBoxes_ == nullptr) {
    cachedBoxes_.reset(new Tensor());
    cachedVariances_.reset(new Tensor());
    cachedBoxes_->mutableData<float>(FP32, param_.outputBoxes->shape());
    cachedVariances_->mutableData<float>(FP32, param_.outputVariances->shape());
    cachedBoxes_->setDataLocation(CPU);
    cachedVariances_->setDataLocation(CPU);
    compute_prior_box();
  }

  param_.outputBoxes->copyFrom(this->cachedBoxes_.get());
  param_.outputVariances->copyFrom(this->cachedVariances_.get());

  param_.outputBoxes->flush();
  param_.outputVariances->flush();
  param_.outputBoxes->setCached(true);
  param_.outputVariances->setCached(true);
  return true;
}

}  // namespace zynqmp
}  // namespace paddle
