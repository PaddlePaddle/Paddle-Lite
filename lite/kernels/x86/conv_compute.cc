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

#include "lite/kernels/x86/conv_compute.h"
#include <utility>
#include "lite/kernels/x86/conv_depthwise.h"
#include "lite/kernels/x86/conv_direct.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace x86 {

template <>
void Conv2dCompute<PRECISION(kFloat), PRECISION(kFloat)>::PrepareForRun() {
  auto& param = this->Param<param_t>();

  const int input_channel = param.x->dims()[1];
  const int output_channel = param.filter->dims()[0];
  const int groups = param.groups;

  const int ih = param.x->dims()[2];
  const int iw = param.x->dims()[3];
  const int chout = param.filter->dims()[0];
  const int chin = param.filter->dims()[1];
  const int kernel_h = param.filter->dims()[2];
  const int kernel_w = param.filter->dims()[3];

  const int stride_h = param.strides[0];
  const int stride_w = param.strides[1];
  auto paddings = *param.paddings;
  auto dilations = *param.dilations;
  bool dw_kernel = (input_channel == groups && output_channel == groups);
  bool ks_equal = (stride_h == stride_w) && (kernel_h == kernel_w);
  bool no_dilation = (dilations[0] == 1) && (dilations[1] == 1);
  bool kps_equal = (paddings[0] == paddings[2]) && ks_equal;
  bool flag_dw_3x3 =
      (kernel_h == 3) && (kernel_w == 3) && (stride_h == 1 || stride_h == 2);
  bool flag_dw_5x5 =
      (kernel_h == 5) && (kernel_w == 5) && (stride_h == 1 || stride_h == 2);
  // todo add conv_5x5_depthwise implement
  flag_dw_5x5 = false;
  bool flag_dw = flag_dw_3x3 || flag_dw_5x5;

  bool nodilations = true;
  for (auto ele : *(param.dilations))
    if (ele != 1) nodilations = false;

  bool paddings_equal = (paddings[0] == paddings[1]) && (paddings[2] == paddings[3]);

  /// select conv impl
  if (dw_kernel && kps_equal && no_dilation && flag_dw && (groups & 3) == 0) {
    impl_ = new DepthwiseConv<PRECISION(kFloat), PRECISION(kFloat)>;
  } else if (chin * chout < 4 * ih * iw && chout % 8 == 0 && groups == 1 &&
             kernel_h == 3 && kernel_w == 3 && nodilations &&
             stride_h == 2 && stride_w == 2 && paddings_equal) {
    impl_ = new DirectConv<float>();
    VLOG(3) << "invoking directConv  3x3s2";
  }

  if (impl_) {
    impl_->SetContext(std::move(this->ctx_));
    impl_->SetParam(param);
    impl_->PrepareForRun();
    is_first_epoch_ = false;
  }
}

template <>
void Conv2dCompute<PRECISION(kFloat), PRECISION(kFloat)>::Run() {
  if (impl_) {
    return impl_->Run();
  }
  // To-do(qili93): remove below lines of code after all kernels implemented
  auto& context = ctx_->As<X86Context>();
  auto& param = *param_.get_mutable<operators::ConvParam>();
  lite::Tensor filter = *param.filter;
  param.output->template mutable_data<float>();
  const int batch_size = static_cast<int>(param.x->dims()[0]);

  const int kh = static_cast<int>(param.filter->dims()[2]);
  const int kw = static_cast<int>(param.filter->dims()[3]);

  const int sh = static_cast<int>(param.strides[0]);
  const int sw = static_cast<int>(param.strides[1]);

  auto paddings = *param.paddings;
  const int ph = paddings[0];
  const int pw = paddings[2];

  bool kps_equal = (pw == ph) && (sw == sh) && (kw == kh);
  bool pads_equal =
      ((paddings[0] == paddings[1]) && (paddings[2] == paddings[3]));
  bool flag_1x1gemm = false;
  if (kw == 1 && sw == 1 && pw == 0 && kps_equal && pads_equal) {
    flag_1x1gemm = true;
  }

  std::vector<int64_t> filter_shape_vec(filter.dims().Vectorize());
  std::vector<int64_t> output_shape_vec(param.output->dims().Vectorize());
  size_t data_dim = filter_shape_vec.size() - 2;
  std::vector<int64_t> col_shape_vec(1 + 2 * data_dim);
  col_shape_vec[0] = param.x->dims()[1] / param.groups;
  for (size_t j = 0; j < data_dim; ++j) {
    col_shape_vec[j + 1] = filter_shape_vec[j + 2];
    col_shape_vec[j + 1 + data_dim] = output_shape_vec[j + 2];
  }
  lite::DDim col_shape(col_shape_vec);
  lite::DDim col_matrix_shape = col_shape.Flatten2D(data_dim + 1);
  bool is_expand = IsExpand(
      filter_shape_vec, param.strides, *param.paddings, *param.dilations);
  lite::Tensor col;
  lite::Tensor col_matrix;
  if (is_expand) {
    col.Resize(col_shape);
    col.mutable_data<float>();
    col_matrix.ShareDataWith(col);
    col_matrix.Resize(col_matrix_shape);
  }
  lite::DDim input_shape = param.x->dims().Slice(1, param.x->dims().size());
  lite::DDim filter_matrix_shape(std::vector<int64_t>{
      filter.dims()[0], filter.dims().production() / filter.dims()[0]});
  filter.Resize(filter_matrix_shape);
  lite::DDim output_matrix_shape(std::vector<int64_t>{
      param.output->dims()[1],
      param.output->dims().production() /
          (param.output->dims()[0] * param.output->dims()[1])});
  int in_step = static_cast<int>(param.x->dims()[1]) / param.groups;
  int out_step = static_cast<int>(param.output->dims()[1]) / param.groups;
  paddle::lite::x86::math::Vol2ColFunctor<lite::TargetType::kX86, float>
      vol2col;
  paddle::lite::x86::math::Im2ColFunctor<
      paddle::lite::x86::math::ColFormat::kCFO,
      lite::TargetType::kX86,
      float>
      im2col;
  auto blas =
      paddle::lite::x86::math::GetBlas<lite::TargetType::kX86, float>(context);
  for (int i = 0; i < batch_size; i++) {
    lite::Tensor in_batch = param.x->template Slice<float>(i, i + 1);
    in_batch.Resize(input_shape);
    lite::Tensor out_batch = param.output->template Slice<float>(i, i + 1);
    out_batch.Resize(output_matrix_shape);
    for (int g = 0; g < param.groups; g++) {
      lite::Tensor in_slice =
          in_batch.Slice<float>(static_cast<int64_t>(g * in_step),
                                static_cast<int64_t>((g + 1) * in_step));
      auto paddings = *param.paddings;
      if (!is_expand) {
        col.ShareDataWith(in_slice);
        col_matrix.ShareDataWith(col);
        col_matrix.Resize(col_matrix_shape);
      } else if (data_dim == 2U && !flag_1x1gemm) {
        // im2col
        im2col(context,
               in_slice,
               *param.dilations,
               param.strides,
               std::vector<int>{
                   paddings[0], paddings[2], paddings[0], paddings[2]},
               &(col));
      } else if (data_dim == 3U) {
        // vol2col
        vol2col(context,
                in_slice,
                *param.dilations,
                param.strides,
                *param.paddings,
                &(col));
      }

      // gemm
      lite::Tensor out_slice;
      out_slice =
          out_batch.Slice<float>(static_cast<int64_t>(g * out_step),
                                 static_cast<int64_t>((g + 1) * out_step));
      lite::Tensor filter_slice;
      filter_slice =
          filter.Slice<float>(static_cast<int64_t>(g * out_step),
                              static_cast<int64_t>((g + 1) * out_step));
      blas.MatMul(
          filter_slice, false, col_matrix, false, 1.0, &(out_slice), 0.0);
    }
  }

  // for bias
  if (param.bias) {
    const int output_channel = static_cast<int>(param.output->dims()[1]);
    const int output_number =
        param.output->dims().production() /
        (param.output->dims()[0] * param.output->dims()[1]);
    auto* bias_data = param.bias->template data<float>();
    auto* out_data = param.output->template mutable_data<float>();
    auto act_param = param.activation_param;
    if (act_param.has_active) {
      if (act_param.active_type == lite_api::ActivationType::kRelu) {
        lite::x86::math::bias_add_relu_broadcast(out_data,
                                                 bias_data,
                                                 out_data,
                                                 batch_size,
                                                 output_channel,
                                                 output_number);
      } else if (act_param.active_type == lite_api::ActivationType::kRelu6) {
        lite::x86::math::bias_add_relu6_broadcast(out_data,
                                                  bias_data,
                                                  out_data,
                                                  batch_size,
                                                  output_channel,
                                                  output_number);
      } else {
        LOG(FATAL) << "[X86] unsupported Activation type";
      }
    } else {
      lite::x86::math::bias_add_broadcast(out_data,
                                          bias_data,
                                          out_data,
                                          batch_size,
                                          output_channel,
                                          output_number);
    }
  }
}

}  // namespace x86
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
typedef paddle::lite::kernels::x86::Conv2dCompute<PRECISION(kFloat),
                                                  PRECISION(kFloat)>
    ConvFp32;

REGISTER_LITE_KERNEL(conv2d, kX86, kFloat, kNCHW, ConvFp32, def)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kFloat))})
    .BindInput("SecondInput",
               {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kFloat))})
    .BindInput("Filter",
               {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kFloat))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kFloat))})
    .BindOutput("Output",
                {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kFloat))})
    .BindPaddleOpVersion("conv2d", 1)
    .Finalize();

REGISTER_LITE_KERNEL(depthwise_conv2d, kX86, kFloat, kNCHW, ConvFp32, def)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kFloat))})
    .BindInput("Filter",
               {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kFloat))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kFloat))})
    .BindOutput("Output",
                {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kFloat))})
    .BindPaddleOpVersion("depthwise_conv2d", 1)
    .Finalize();
