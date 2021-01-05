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

#include "lite/kernels/xpu/interpolate_compute.h"
#include <iostream>
#include <memory>
#include <vector>
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

inline std::vector<int> get_new_shape(
    std::vector<const lite::Tensor*> list_new_shape_tensor) {
  // get tensor from
  std::vector<int> vec_new_shape;
  for (size_t i = 0; i < list_new_shape_tensor.size(); ++i) {
    auto tensor = list_new_shape_tensor[i];
    vec_new_shape.push_back(static_cast<int32_t>(*tensor->data<int32_t>()));
  }

  return vec_new_shape;
}

template <typename T>
inline std::vector<T> get_new_data_from_tensor(const Tensor* new_data_tensor) {
  std::vector<T> vec_new_data;
  auto* new_data = new_data_tensor->data<T>();
  lite::Tensor cpu_starts_tensor;
  vec_new_data =
      std::vector<T>(new_data, new_data + new_data_tensor->dims().production());
  return vec_new_data;
}

void PrepareLayout(lite::Tensor* input,
                   lite::Tensor* out_size,
                   std::vector<const lite::Tensor*> list_new_size_tensor,
                   lite::Tensor* scale_tensor,
                   lite::Tensor* output,
                   float scale,
                   int* out_h,
                   int* out_w) {
  // format NCHW
  int n = input->dims()[0];
  int c = input->dims()[1];
  int in_h = input->dims()[2];
  int in_w = input->dims()[3];
  if (list_new_size_tensor.size() > 0) {
    // have size tensor
    auto new_size = get_new_shape(list_new_size_tensor);
    *out_h = new_size[0];
    *out_w = new_size[1];
  } else {
    if (scale_tensor != nullptr) {
      auto scale_data = get_new_data_from_tensor<float>(scale_tensor);
      scale = scale_data[0];
    }
    if (scale > 0) {
      *out_h = static_cast<int>(in_h * scale);
      *out_w = static_cast<int>(in_w * scale);
    }
    if (out_size != nullptr) {
      auto out_size_data = get_new_data_from_tensor<int>(out_size);
      *out_h = out_size_data[0];
      *out_w = out_size_data[1];
    }
  }
  output->Resize({n, c, *out_h, *out_w});
}

void BilinearInterpCompute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->As<XPUContext>();
  lite::Tensor* X = param.X;
  int n = X->dims()[0];
  int c = X->dims()[1];
  int in_h = X->dims()[2];
  int in_w = X->dims()[3];

  // optionla inputs
  lite::Tensor* OutSize = param.OutSize;
  auto SizeTensor = param.SizeTensor;
  auto Scale = param.Scale;
  // output
  lite::Tensor* Out = param.Out;
  // optional attributes
  float scale = param.scale;
  int out_w = param.out_w;
  int out_h = param.out_h;
  int align_mode = param.align_mode;
  // required attributes
  bool align_corners = param.align_corners;
  PrepareLayout(X, OutSize, SizeTensor, Scale, Out, scale, &out_h, &out_w);

  int trans_mode = -1;
  if (align_corners == true) {
    trans_mode = 0;
  } else if ((align_corners == false) && (align_mode == 0)) {
    trans_mode = 1;
  } else {
    trans_mode = 2;
  }
  int r = xdnn::interpolate2d<float>(ctx.GetRawContext(), /* context */
                                     X->data<float>(),
                                     Out->mutable_data<float>(TARGET(kXPU)),
                                     n,
                                     c,
                                     in_h,
                                     in_w,
                                     out_h,
                                     out_w,
                                     false,
                                     trans_mode,
                                     true);
  CHECK_EQ(r, 0);
}

void NearestInterpCompute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->As<XPUContext>();
  lite::Tensor* X = param.X;
  int n = X->dims()[0];
  int c = X->dims()[1];
  int in_h = X->dims()[2];
  int in_w = X->dims()[3];

  // optionla inputs
  lite::Tensor* OutSize = param.OutSize;
  auto SizeTensor = param.SizeTensor;
  auto Scale = param.Scale;
  // output
  lite::Tensor* Out = param.Out;
  // optional attributes
  float scale = param.scale;
  int out_w = param.out_w;
  int out_h = param.out_h;
  // required attributes
  bool align_corners = param.align_corners;
  PrepareLayout(X, OutSize, SizeTensor, Scale, Out, scale, &out_h, &out_w);
  int trans_mode = (align_corners == true) ? 0 : 1;

  int r = xdnn::interpolate2d<float>(ctx.GetRawContext(), /* context */
                                     X->data<float>(),
                                     Out->mutable_data<float>(TARGET(kXPU)),
                                     n,
                                     c,
                                     in_h,
                                     in_w,
                                     out_h,
                                     out_w,
                                     true,
                                     trans_mode,
                                     true);

  CHECK_EQ(r, 0);
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(bilinear_interp,
                     kXPU,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::xpu::BilinearInterpCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("OutSize",
               {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt32))})
    .BindInput("SizeTensor",
               {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt32))})
    .BindInput("Scale", {LiteType::GetTensorTy(TARGET(kX86))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();

REGISTER_LITE_KERNEL(nearest_interp,
                     kXPU,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::xpu::NearestInterpCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("OutSize",
               {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt32))})
    .BindInput("SizeTensor",
               {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt32))})
    .BindInput("Scale", {LiteType::GetTensorTy(TARGET(kX86))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();
