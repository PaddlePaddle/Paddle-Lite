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

#include <memory>
#include <vector>
#include "lite/backends/cuda/math/gemm.h"
#include "lite/core/op_registry.h"
#include "lite/core/target_wrapper.h"
#include "lite/core/tensor.h"
#include "lite/kernels/cuda/var_conv_2d_compute.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace cuda {

inline int ConvOutputSize(int input_size,
                          int filter_size,
                          int dilation,
                          int pad_left,
                          int pad_right,
                          int stride) {
  const int dkernel = dilation * (filter_size - 1) + 1;
  int output_size =
      (input_size + (pad_left + pad_right) - dkernel) / stride + 1;

  return output_size;
}

void VarConv2DCompute::PrepareForRun() {
  auto& context = this->ctx_->template As<CUDAContext>();
  auto stream = context.exec_stream();
  auto& param = this->Param<param_t>();
  conv_param_.x = const_cast<lite::Tensor*>(param.X);
  conv_param_.var_length = true;

  conv_param_.paddings.reset(new std::vector<int>);
  conv_param_.paddings->push_back(static_cast<int>(param.kernel_h / 2));
  conv_param_.paddings->push_back(static_cast<int>(param.kernel_h / 2));
  conv_param_.paddings->push_back(static_cast<int>(param.kernel_w / 2));
  conv_param_.paddings->push_back(static_cast<int>(param.kernel_w / 2));
  conv_param_.dilations.reset(new std::vector<int>);
  conv_param_.dilations->push_back(1);
  conv_param_.dilations->push_back(1);
  conv_param_.strides[0] = param.stride_h;
  conv_param_.strides[1] = param.stride_w;
  conv_param_.filter = const_cast<lite::Tensor*>(param.W);
  conv_param_.filter->Resize({param.output_channel,
                              param.input_channel,
                              param.kernel_h,
                              param.kernel_w});

  conv_param_.output = param.Out;
  std::vector<int64_t> output_shape(
      {conv_param_.x->dims()[0], param.output_channel});
  for (size_t i = 0; i < conv_param_.strides.size(); ++i) {
    output_shape.push_back(
        ConvOutputSize(conv_param_.x->dims()[i + 2],
                       conv_param_.filter->dims()[i + 2],
                       (*conv_param_.dilations.get())[i],
                       (*conv_param_.paddings.get())[i * 2],
                       (*conv_param_.paddings.get())[i * 2 + 1],
                       conv_param_.strides[i]));
  }
  if (param.fuse_relu) {
    conv_param_.activation_param.has_active = true;
    conv_param_.activation_param.active_type = lite_api::ActivationType::kRelu;
  }
  conv_param_.output->Resize({output_shape});
  conv_impl_.reset(new lite::cuda::math::CudnnConv2D<PRECISION(kFloat)>);
  conv_impl_->init(conv_param_, &context);
}

void VarConv2DCompute::Run() {
  auto& context = this->ctx_->template As<CUDAContext>();
  auto stream = context.exec_stream();
  auto& param = this->Param<param_t>();

  param.Out->set_lod(param.X->lod());
  std::vector<int64_t> output_shape(
      {conv_param_.x->dims()[0], param.output_channel});
  for (size_t i = 0; i < conv_param_.strides.size(); ++i) {
    output_shape.push_back(
        ConvOutputSize(conv_param_.x->dims()[i + 2],
                       conv_param_.filter->dims()[i + 2],
                       (*conv_param_.dilations.get())[i],
                       (*conv_param_.paddings.get())[i * 2],
                       (*conv_param_.paddings.get())[i * 2 + 1],
                       conv_param_.strides[i]));
  }
  conv_param_.output->Resize({output_shape});
  conv_impl_->create(conv_param_, &context);
  conv_impl_->run(conv_param_);
}

}  // namespace cuda
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(var_conv_2d,
                     kCUDA,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::cuda::VarConv2DCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kCUDA))})
    .BindInput("W", {LiteType::GetTensorTy(TARGET(kCUDA))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kCUDA))})
    .BindOutput("Col", {LiteType::GetTensorTy(TARGET(kCUDA))})
    .Finalize();
