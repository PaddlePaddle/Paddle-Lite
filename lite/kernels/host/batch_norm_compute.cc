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

#include "lite/kernels/host/batch_norm_compute.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace host {
void BatchNormCompute::Run() {
  auto& param = Param<operators::BatchNormParam>();
  int channel_size = 0;
  if (data_layout_ == "NCHW") {
    channel_size = dims_[1];
  } else {
    LOG(FATAL) << "Unknown storage order: " << data_layout_;
  }

  auto x_data = param.x->data<float>();
  auto y_data = param.y->mutable_data<float>();
  auto scale_data = param.scale->data<float>();
  auto bias_data = param.bias->data<float>();
  auto mean_data = param.mean->data<float>();
  auto variance_data = param.variance->data<float>();

  int outer_size = 0;
  int inner_size = 0;
  if (data_layout_ == "NCHW") {
    outer_size = dims_[0];
    inner_size = dims_.Slice(2, dims_.size()).production();
  } else {
    LOG(FATAL) << "Unknown storage order: " << data_layout_;
  }
  auto x_ptr = x_data;
  auto y_ptr = y_data;
  for (int o = 0; o < outer_size; o++) {
    for (int c = 0; c < channel_size; c++) {
      for (int i = 0; i < inner_size; i++) {
        float norm_x =
            (*x_ptr - mean_data[c]) / std::sqrt(variance_data[c] + epsilon_);
        *y_ptr = norm_x * scale_data[c] + bias_data[c];
        x_ptr++;
        y_ptr++;
      }
    }
  }
}

}  // namespace host
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(batch_norm,
                     kHost,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::arm::BatchNormCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindInput("Scale", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindInput("Mean", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindInput("Variance", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("Y", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("MeanOut", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("VarianceOut", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("SavedMean", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("SavedVariance", {LiteType::GetTensorTy(TARGET(kHost))})
    .Finalize();
