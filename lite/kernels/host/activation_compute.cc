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

#include "lite/kernels/host/activation_compute.h"
#include <cmath>
namespace paddle {
namespace lite {
namespace kernels {
namespace host {

void ReluCompute::Run() {
  auto& param = this->Param<param_t>();
  CHECK(param.X);
  auto x_dims = param.X->dims();
  auto x_data = param.X->data<float>();
  auto output_data = param.Out->mutable_data<float>();
  for (int i = 0; i < x_dims.production(); i++) {
    output_data[i] = x_data[i] > 0 ? x_data[i] : 0;
  }
}

void LeakyReluCompute::Run() {
  auto& param = this->Param<param_t>();
  CHECK(param.X);
  auto x_dims = param.X->dims();
  auto x_data = param.X->data<float>();
  auto alpha = param.Leaky_relu_alpha;
  auto output_data = param.Out->mutable_data<float>();
  for (int i = 0; i < x_dims.production(); i++) {
    output_data[i] = x_data[i] > 0 ? x_data[i] : x_data[i] * alpha;
  }
}

void ReluClippedCompute::Run() {
  auto& param = this->Param<param_t>();
  CHECK(param.X);
  auto x_dims = param.X->dims();
  auto x_data = param.X->data<float>();
  auto coef = param.Relu_clipped_coef;
  auto output_data = param.Out->mutable_data<float>();
  for (int i = 0; i < x_dims.production(); i++) {
    output_data[i] = x_data[i] > 0 ? x_data[i] : 0;
    output_data[i] = output_data[i] < coef ? output_data[i] : coef;
  }
}

void PReluCompute::Run() {
  auto& param = this->Param<param_t>();
  CHECK(param.X);
  auto x_dims = param.X->dims();
  auto x_data = param.X->data<float>();
  auto mode = param.Prelu_mode;
  auto alpha_data = param.Prelu_alpha->data<float>();
  auto output_data = param.Out->mutable_data<float>();

  int outer_size = x_dims[0];
  int channel_size = x_dims[1];
  int inner_size = x_dims.count(2, x_dims.size());
  if (mode == "all" || mode == "channel") {
    int stride_size = inner_size * channel_size;
    for (int n = 0; n < outer_size; n++) {
      const float* data_in_batch = x_data + n * stride_size;
      float* data_out_batch = output_data + n * stride_size;
      for (int c = 0; c < channel_size; c++) {
        const float* data_in_c = data_in_batch + c * inner_size;
        float* data_out_c = data_out_batch + c * inner_size;
        float slope = mode == "all" ? alpha_data[0] : alpha_data[c];
        for (int i = 0; i < inner_size; i++) {
          data_out_c[i] =
              data_in_c[i] > 0.f ? data_in_c[i] : data_in_c[i] * slope;
        }
      }
    }
  } else {  // mode = element
    for (int i = 0; i < x_dims.production(); i++) {
      output_data[i] = x_data[i] > 0.f ? x_data[i] : x_data[i] * alpha_data[i];
    }
  }
}

void SigmoidCompute::Run() {
  auto& param = this->Param<param_t>();
  CHECK(param.X);
  auto x_dims = param.X->dims();
  auto x_data = param.X->data<float>();
  auto output_data = param.Out->mutable_data<float>();
  for (int i = 0; i < x_dims.production(); i++) {
    output_data[i] = 1 / (1 + std::exp(-x_data[i]));
  }
}

void TanhCompute::Run() {
  auto& param = this->Param<param_t>();
  CHECK(param.X);
  auto x_dims = param.X->dims();
  auto x_data = param.X->data<float>();
  auto output_data = param.Out->mutable_data<float>();
  for (int i = 0; i < x_dims.production(); i++) {
    float x_tmp = x_data[i];
    x_tmp = std::min(x_tmp, 70.00008f);
    x_tmp = std::max(x_tmp, -70.00008f);
    output_data[i] = (std::exp(x_tmp) - std::exp(-x_tmp)) /
                     (std::exp(x_tmp) + std::exp(-x_tmp));
  }
}

void SwishCompute::Run() {
  auto& param = this->Param<param_t>();
  CHECK(param.X);
  auto x_dims = param.X->dims();
  auto x_data = param.X->data<float>();
  auto beta = param.Swish_beta;
  auto output_data = param.Out->mutable_data<float>();
  for (int i = 0; i < x_dims.production(); i++) {
    output_data[i] = x_data[i] / (1 + std::exp(-x_data[i] * beta));
  }
}

void Relu6Compute::Run() {
  auto& param = this->Param<param_t>();
  CHECK(param.X);
  auto x_dims = param.X->dims();
  auto x_data = param.X->data<float>();
  float coef = 6.;
  auto output_data = param.Out->mutable_data<float>();
  for (int i = 0; i < x_dims.production(); i++) {
    output_data[i] = x_data[i] > 0 ? x_data[i] : 0;
    output_data[i] = output_data[i] < coef ? output_data[i] : coef;
  }
}

void LogCompute::Run() {
  auto& param = this->Param<param_t>();
  CHECK(param.X);
  auto x_dims = param.X->dims();
  auto x_data = param.X->data<float>();
  auto output_data = param.Out->mutable_data<float>();
  for (int i = 0; i < x_dims.production(); i++) {
    output_data[i] = std::log(x_data[i]);
  }
}

void ExpCompute::Run() {
  auto& param = this->Param<param_t>();
  CHECK(param.X);
  auto x_dims = param.X->dims();
  auto x_data = param.X->data<float>();
  auto output_data = param.Out->mutable_data<float>();
  for (int i = 0; i < x_dims.production(); i++) {
    output_data[i] = std::exp(x_data[i]);
  }
}

void FloorCompute::Run() {
  auto& param = this->Param<param_t>();
  CHECK(param.X);
  auto x_dims = param.X->dims();
  auto x_data = param.X->data<float>();
  auto output_data = param.Out->mutable_data<float>();
  for (int i = 0; i < x_dims.production(); i++) {
    output_data[i] = std::floor(x_data[i]);
  }
}

void HardSigmoidCompute::Run() {
  auto& param = this->Param<param_t>();
  CHECK(param.X);
  auto x_dims = param.X->dims();
  auto x_data = param.X->data<float>();
  float slope = param.hard_sigmoid_slope;
  float offset = param.hard_sigmoid_offset;
  auto output_data = param.Out->mutable_data<float>();
  for (int i = 0; i < x_dims.production(); i++) {
    float tmp = x_data[i] * slope + offset;
    tmp = tmp < 1.0f ? tmp : 1.0f;
    tmp = tmp > 0.0f ? tmp : 0.0f;
    output_data[i] = tmp;
  }
}

void RsqrtCompute::Run() {
  auto& param = this->Param<param_t>();
  CHECK(param.X);
  auto x_dims = param.X->dims();
  auto x_data = param.X->data<float>();
  auto output_data = param.Out->mutable_data<float>();
  for (int i = 0; i < x_dims.production(); i++) {
    output_data[i] = 1.0 / std::sqrt(x_data[i]);
  }
}

void SquareCompute::Run() {
  auto& param = this->Param<param_t>();
  CHECK(param.X);
  auto x_dims = param.X->dims();
  auto x_data = param.X->data<float>();
  auto output_data = param.Out->mutable_data<float>();
  for (int i = 0; i < x_dims.production(); i++) {
    output_data[i] = x_data[i] * x_data[i];
  }
}

void HardSwishCompute::Run() {
  auto& param = this->Param<param_t>();
  CHECK(param.X);
  auto x_dims = param.X->dims();
  auto x_data = param.X->data<float>();
  auto output_data = param.Out->mutable_data<float>();
  float threshold = param.hard_swish_threshold;
  float scale = param.hard_swish_scale;
  float offset = param.hard_swish_offset;
  for (int i = 0; i < x_dims.production(); i++) {
    output_data[i] =
        (std::min)((std::max)(0.f, x_data[i] + offset), threshold) * x_data[i] /
        scale;
  }
}

void ReciprocalCompute::Run() {
  auto& param = this->Param<param_t>();
  CHECK(param.X);
  auto x_dims = param.X->dims();
  auto x_data = param.X->data<float>();
  auto output_data = param.Out->mutable_data<float>();
  for (int i = 0; i < x_dims.production(); i++) {
    output_data[i] = 1.0 / x_data[i];
  }
}

void AbsCompute::Run() {
  auto& param = this->Param<param_t>();
  CHECK(param.X);
  auto x_dims = param.X->dims();
  auto x_data = param.X->data<float>();
  auto output_data = param.Out->mutable_data<float>();
  for (int i = 0; i < x_dims.production(); i++) {
    output_data[i] = (x_data[i] > 0.f ? x_data[i] : -x_data[i]);
  }
}

void ThresholdedReluCompute::Run() {
  auto& param = this->Param<param_t>();
  CHECK(param.X);
  auto x_dims = param.X->dims();
  auto x_data = param.X->data<float>();
  auto output_data = param.Out->mutable_data<float>();
  float threshold = param.relu_threshold;
  for (int i = 0; i < x_dims.production(); i++) {
    output_data[i] = (x_data[i] > threshold ? x_data[i] : 0.f);
  }
}

void EluCompute::Run() {
  auto& param = this->Param<param_t>();
  CHECK(param.X);
  auto x_dims = param.X->dims();
  auto x_data = param.X->data<float>();
  auto output_data = param.Out->mutable_data<float>();
  float alpha = param.Elu_alpha;
  for (int i = 0; i < x_dims.production(); i++) {
    output_data[i] =
        (x_data[i] < 0) ? (alpha * (std::exp(x_data[i]) - 1)) : x_data[i];
  }
}

void SoftplusCompute::Run() {
  auto& param = this->Param<param_t>();
  CHECK(param.X);
  auto x_dims = param.X->dims();
  auto x_data = param.X->data<float>();
  auto output_data = param.Out->mutable_data<float>();
  float beta = param.softplus_beta;
  float threshold = param.softplus_threshold;
  for (int i = 0; i < x_dims.production(); i++) {
    output_data[i] = x_data[i] * beta > threshold
                         ? x_data[i]
                         : std::log(1 + std::exp(x_data[i] * beta)) / beta;
  }
}

}  // namespace host
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(
    relu, kHost, kFloat, kNCHW, paddle::lite::kernels::host::ReluCompute, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kHost))})
    .Finalize();
REGISTER_LITE_KERNEL(leaky_relu,
                     kHost,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::host::LeakyReluCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindInput("alpha", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kHost))})
    .Finalize();
REGISTER_LITE_KERNEL(relu_clipped,
                     kHost,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::host::ReluClippedCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindInput("Relu_clipped_coef", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kHost))})
    .Finalize();
REGISTER_LITE_KERNEL(
    prelu, kHost, kFloat, kNCHW, paddle::lite::kernels::host::PReluCompute, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindInput("mode", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindInput("Alpha", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kHost))})
    .Finalize();
REGISTER_LITE_KERNEL(sigmoid,
                     kHost,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::host::SigmoidCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kHost))})
    .Finalize();
REGISTER_LITE_KERNEL(
    tanh, kHost, kFloat, kNCHW, paddle::lite::kernels::host::TanhCompute, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kHost))})
    .Finalize();
REGISTER_LITE_KERNEL(
    swish, kHost, kFloat, kNCHW, paddle::lite::kernels::host::SwishCompute, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindInput("beta", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kHost))})
    .Finalize();
REGISTER_LITE_KERNEL(
    relu6, kHost, kFloat, kNCHW, paddle::lite::kernels::host::Relu6Compute, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kHost))})
    .Finalize();
REGISTER_LITE_KERNEL(
    log, kHost, kFloat, kNCHW, paddle::lite::kernels::host::LogCompute, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kHost))})
    .Finalize();
REGISTER_LITE_KERNEL(
    exp, kHost, kFloat, kNCHW, paddle::lite::kernels::host::ExpCompute, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kHost))})
    .Finalize();
REGISTER_LITE_KERNEL(
    floor, kHost, kFloat, kNCHW, paddle::lite::kernels::host::FloorCompute, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kHost))})
    .Finalize();
REGISTER_LITE_KERNEL(hard_sigmoid,
                     kHost,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::host::HardSigmoidCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kHost))})
    .Finalize();
REGISTER_LITE_KERNEL(
    rsqrt, kHost, kFloat, kNCHW, paddle::lite::kernels::host::RsqrtCompute, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kHost))})
    .Finalize();
REGISTER_LITE_KERNEL(square,
                     kHost,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::host::SquareCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kHost))})
    .Finalize();
REGISTER_LITE_KERNEL(hard_swish,
                     kHost,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::host::HardSwishCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kHost))})
    .Finalize();
REGISTER_LITE_KERNEL(reciprocal,
                     kHost,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::host::ReciprocalCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kHost))})
    .Finalize();
REGISTER_LITE_KERNEL(
    abs, kHost, kFloat, kNCHW, paddle::lite::kernels::host::AbsCompute, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kHost))})
    .Finalize();
REGISTER_LITE_KERNEL(thresholded_relu,
                     kHost,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::host::ThresholdedReluCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kHost))})
    .Finalize();
REGISTER_LITE_KERNEL(
    elu, kHost, kFloat, kNCHW, paddle::lite::kernels::host::EluCompute, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kHost))})
    .Finalize();
REGISTER_LITE_KERNEL(softplus,
                     kHost,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::host::SoftplusCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kHost))})
    .Finalize();
