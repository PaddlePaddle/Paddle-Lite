/* Copyright (c) 2016 Baidu, Inc. All Rights Reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
==============================================================================*/

#pragma once

#include "operators/kernel/box_coder_kernel.h"

namespace paddle_mobile {
namespace operators {

template <typename T>
void EncodeCenterSize(const framework::Tensor& target_box,
                      const framework::Tensor& prior_box,
                      const framework::Tensor& prior_box_var, T* output) {
  int64_t row = target_box.dims()[0];
  int64_t col = prior_box.dims()[0];
  int64_t len = prior_box.dims()[1];
  auto* target_box_data = target_box.data<T>();
  auto* prior_box_data = prior_box.data<T>();
  auto* prior_box_var_data = prior_box_var.data<T>();

  for (int64_t i = 0; i < row; ++i) {
    for (int64_t j = 0; j < col; ++j) {
      T prior_box_width = prior_box_data[j * len + 2] - prior_box_data[j * len];
      T prior_box_height =
          prior_box_data[j * len + 3] - prior_box_data[j * len + 1];
      T prior_box_center_x =
          (prior_box_data[j * len + 2] + prior_box_data[j * len]) / 2;
      T prior_box_center_y =
          (prior_box_data[j * len + 3] + prior_box_data[j * len + 1]) / 2;

      T target_box_center_x =
          (target_box_data[i * len + 2] + target_box_data[i * len]) / 2;
      T target_box_center_y =
          (target_box_data[i * len + 3] + target_box_data[i * len + 1]) / 2;
      T target_box_width =
          target_box_data[i * len + 2] - target_box_data[i * len];
      T target_box_height =
          target_box_data[i * len + 3] - target_box_data[i * len + 1];

      size_t offset = i * col * len + j * len;
      output[offset] = (target_box_center_x - prior_box_center_x) /
                       prior_box_width / prior_box_var_data[j * len];
      output[offset + 1] = (target_box_center_y - prior_box_center_y) /
                           prior_box_height / prior_box_var_data[j * len + 1];
      output[offset + 2] =
          std::log(std::fabs(target_box_width / prior_box_width)) /
          prior_box_var_data[j * len + 2];
      output[offset + 3] =
          std::log(std::fabs(target_box_height / prior_box_height)) /
          prior_box_var_data[j * len + 3];
    }
  }
}

template <typename T>
void DecodeCenterSize(const framework::Tensor& target_box,
                      const framework::Tensor& prior_box,
                      const framework::Tensor& prior_box_var, T* output) {
  int64_t row = target_box.dims()[0];
  int64_t col = prior_box.dims()[0];
  int64_t len = prior_box.dims()[1];

  auto* target_box_data = target_box.data<T>();
  auto* prior_box_data = prior_box.data<T>();
  auto* prior_box_var_data = prior_box_var.data<T>();

  for (int64_t i = 0; i < row; ++i) {
    for (int64_t j = 0; j < col; ++j) {
      size_t offset = i * col * len + j * len;
      T prior_box_width = prior_box_data[j * len + 2] - prior_box_data[j * len];
      T prior_box_height =
          prior_box_data[j * len + 3] - prior_box_data[j * len + 1];
      T prior_box_center_x =
          (prior_box_data[j * len + 2] + prior_box_data[j * len]) / 2;
      T prior_box_center_y =
          (prior_box_data[j * len + 3] + prior_box_data[j * len + 1]) / 2;

      T target_box_center_x = prior_box_var_data[j * len] *
                                  target_box_data[offset] * prior_box_width +
                              prior_box_center_x;
      T target_box_center_y = prior_box_var_data[j * len + 1] *
                                  target_box_data[offset + 1] *
                                  prior_box_height +
                              prior_box_center_y;
      T target_box_width = std::exp(prior_box_var_data[j * len + 2] *
                                    target_box_data[offset + 2]) *
                           prior_box_width;
      T target_box_height = std::exp(prior_box_var_data[j * len + 3] *
                                     target_box_data[offset + 3]) *
                            prior_box_height;

      output[offset] = target_box_center_x - target_box_width / 2;
      output[offset + 1] = target_box_center_y - target_box_height / 2;
      output[offset + 2] = target_box_center_x + target_box_width / 2;
      output[offset + 3] = target_box_center_y + target_box_height / 2;
    }
  }
}

template <>
void BoxCoderKernel<CPU, float>::Compute(const BoxCoderParam& param) const {
  const auto* input_priorbox = param.InputPriorBox();
  const auto* input_priorboxvar = param.InputPriorBoxVar();
  const auto* input_targetbox = param.InputTargetBox();

  const auto& code_type = param.CodeType();

  auto row = input_targetbox->dims()[0];
  auto col = input_priorbox->dims()[0];
  auto len = input_priorbox->dims()[1];

  Tensor* output_box = param.OutputBox();
  auto* output_box_dataptr = output_box->mutable_data<float>({row, col, len});

  if (code_type == "encode_center_size") {
    EncodeCenterSize<float>(*input_targetbox, *input_priorbox,
                            *input_priorboxvar, output_box_dataptr);
  }
  if (code_type == "decode_center_size") {
    DecodeCenterSize<float>(*input_targetbox, *input_priorbox,
                            *input_priorboxvar, output_box_dataptr);
  }
}
}  // namespace operators
}  // namespace paddle_mobile
