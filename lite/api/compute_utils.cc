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

#include "compute_utils.h"  // NOLINT
#include "lite/backends/arm/math/type_trans.h"
#include "lite/core/tensor.h"
#include "log_lite.h"      // NOLINT
#include "paddle_place.h"  // NOLINT
namespace paddle {
namespace lite_api {

// clang-format off
void ComputeUtils::TensorFloatToInt8(Tensor& tin, Tensor& tout, float scale) {
  lite::Tensor* raw_tin = static_cast<lite::Tensor*>(tin.GetRawTensor());
  lite::Tensor* raw_tout = static_cast<lite::Tensor*>(tout.GetRawTensor());
  LCHECK(raw_tin, "tensor in must have raw tensor\n");
  tout.Resize(tin.shape());
  int outer_size = 1;
  int axis_size = 1;
  int inner_size = raw_tin->numel();
  const float* din = raw_tin->data<float>();
  int8_t* dout = raw_tout->mutable_data<int8_t>();
  paddle::lite::arm::math::fp32_to_int8(
      din, dout, &scale, axis_size, outer_size, inner_size);
}

void ComputeUtils::TensorFloatToInt8Inplace(Tensor& tin, float scale) {
  lite::Tensor* raw_tin = static_cast<lite::Tensor*>(tin.GetRawTensor());
  LCHECK(raw_tin, "tensor in must have raw tensor\n");
  LCHECK_GT(raw_tin->numel(), 0, "tensor in shape must greater than zero\n");
  LCHECK_EQ(raw_tin->precision(),
            PRECISION(kFloat),
            "tensor in precision must be float\n");
  int outer_size = 1;
  int axis_size = 1;
  int inner_size = raw_tin->numel();
  const float* din = raw_tin->data<float>();
  int8_t* dout = raw_tin->mutable_data<int8_t>();
  paddle::lite::arm::math::fp32_to_int8(
      din, dout, &scale, axis_size, outer_size, inner_size);
}

void ComputeUtils::TensorInt8ToFloat(Tensor& tin, Tensor& tout, float scale) {
  lite::Tensor* raw_tin = static_cast<lite::Tensor*>(tin.GetRawTensor());
  lite::Tensor* raw_tout = static_cast<lite::Tensor*>(tout.GetRawTensor());
  LCHECK(raw_tin, "tensor in must have raw tensor\n");
  LCHECK_GT(raw_tin->numel(), 0, "tensor in shape must greater than zero\n");
  LCHECK_EQ(raw_tin->precision(),
            PRECISION(kInt8),
            "tensor in precision must be int8");
  tout.Resize(tin.shape());
  int outer_size = 1;
  int axis_size = 1;
  int inner_size = raw_tin->numel();
  const int8_t* din = raw_tin->data<int8_t>();
  float* dout = raw_tout->mutable_data<float>();
  paddle::lite::arm::math::int8_to_fp32(
      din, dout, &scale, axis_size, outer_size, inner_size);
}

void ComputeUtils::TensorInt8ToFloatInplace(Tensor& tin, float scale) {
  lite::Tensor* raw_tin = static_cast<lite::Tensor*>(tin.GetRawTensor());
  lite::Tensor tmp_out;
  LCHECK(raw_tin, "tensor in must have raw tensor\n");
  LCHECK_GT(raw_tin->numel(), 0, "tensor in shape must greater than zero\n");
  LCHECK_EQ(raw_tin->precision(),
            PRECISION(kInt8),
            "tensor in precision must be int8");
  tmp_out.Resize(tin.shape());
  int outer_size = 1;
  int axis_size = 1;
  int inner_size = raw_tin->numel();
  const int8_t* din = raw_tin->data<int8_t>();
  float* tmp_dout = tmp_out.mutable_data<float>();
  paddle::lite::arm::math::int8_to_fp32(
      din, tmp_dout, &scale, axis_size, outer_size, inner_size);
  float* dout = raw_tin->mutable_data<float>();
  memcpy(dout, tmp_dout, raw_tin->numel() * sizeof(float));
}

void ComputeUtils::ConvWeightsFloatToInt8(Tensor& weightin,
                                          Tensor& weightout,
                                          std::vector<float> scale) {
  lite::Tensor* raw_win = static_cast<lite::Tensor*>(weightin.GetRawTensor());
  lite::Tensor* raw_wout = static_cast<lite::Tensor*>(weightout.GetRawTensor());
  LCHECK(raw_win, "weights in must have raw tensor\n");
  LCHECK_GT(raw_win->numel(), 0, "weights in shape must greater than zero\n");
  LCHECK_EQ(raw_win->precision(),
            PRECISION(kFloat),
            "weights in precision must be float");
  weightout.Resize(weightin.shape());
  int outer_size = 1;
  int axis_size = raw_win->dims()[0];  // chout
  int inner_size =
      raw_win->numel() / axis_size;  // chin / group * ksize_w * ksize_h
  const float* din = raw_win->data<float>();
  int8_t* dout = raw_wout->mutable_data<int8_t>();
  paddle::lite::arm::math::fp32_to_int8(
      din, dout, scale.data(), axis_size, outer_size, inner_size);
}

void ComputeUtils::ConvWeightsFloatToInt8Inplace(Tensor& weightin,
                                                 std::vector<float> scale) {
  lite::Tensor* raw_win = static_cast<lite::Tensor*>(weightin.GetRawTensor());
  LCHECK(raw_win, "weights in must have raw tensor\n");
  LCHECK_GT(raw_win->numel(), 0, "weights in shape must greater than zero\n");
  LCHECK_EQ(raw_win->precision(),
            PRECISION(kFloat),
            "weights in precision must be float");
  int outer_size = 1;
  int axis_size = raw_win->dims()[0];  // chout
  int inner_size =
      raw_win->numel() / axis_size;  // chin / group * ksize_w * ksize_h
  const float* din = raw_win->data<float>();
  int8_t* dout = raw_win->mutable_data<int8_t>();
  paddle::lite::arm::math::fp32_to_int8(
      din, dout, scale.data(), axis_size, outer_size, inner_size);
}

void ComputeUtils::ConvWeightsInt8ToFloat(Tensor& weightin,
                                          Tensor& weightout,
                                          std::vector<float> scale) {
  lite::Tensor* raw_win = static_cast<lite::Tensor*>(weightin.GetRawTensor());
  lite::Tensor* raw_wout = static_cast<lite::Tensor*>(weightout.GetRawTensor());
  LCHECK(raw_win, "weights in must have raw tensor\n");
  LCHECK_GT(raw_win->numel(), 0, "weights in shape must greater than zero\n");
  LCHECK_EQ(raw_win->precision(),
            PRECISION(kInt8),
            "weights in precision must be int8");
  weightout.Resize(weightin.shape());
  int outer_size = 1;
  int axis_size = raw_win->dims()[0];  // chout
  int inner_size =
      raw_win->numel() / axis_size;  // chin / group * ksize_w * ksize_h
  const int8_t* din = raw_win->data<int8_t>();
  float* dout = raw_wout->mutable_data<float>();
  paddle::lite::arm::math::int8_to_fp32(
      din, dout, scale.data(), axis_size, outer_size, inner_size);
}

void ComputeUtils::ConvWeightsInt8ToFloatInplace(Tensor& weightin,
                                                 std::vector<float> scale) {
  lite::Tensor* raw_win = static_cast<lite::Tensor*>(weightin.GetRawTensor());
  lite::Tensor tmp_out;
  LCHECK(raw_win, "weights in must have raw tensor\n");
  LCHECK_GT(raw_win->numel(), 0, "weights in shape must greater than zero\n");
  LCHECK_EQ(raw_win->precision(),
            PRECISION(kInt8),
            "weights in precision must be int8");
  tmp_out.Resize(weightin.shape());
  int outer_size = 1;
  int axis_size = raw_win->dims()[0];  // chout
  int inner_size =
      raw_win->numel() / axis_size;  // chin / group * ksize_w * ksize_h
  const int8_t* din = raw_win->data<int8_t>();
  float* dout_tmp = tmp_out.mutable_data<float>();
  paddle::lite::arm::math::int8_to_fp32(
      din, dout_tmp, scale.data(), axis_size, outer_size, inner_size);
  float* dout = raw_win->mutable_data<float>();
  memcpy(dout, dout_tmp, raw_win->numel() * sizeof(float));
}
// clang-format on

}  // namespace lite_api
}  // namespace paddle
