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
#include "lite/backends/x86/math/fill_bias_activate.h"
#include "lite/kernels/x86/conv_depthwise.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace x86 {
#define INIT_PARAM                      \
  auto& param = this->Param<param_t>(); \
  auto x_dims = param.x->dims();        \
  auto w_dims = param.filter->dims();   \
  auto o_dims = param.output->dims();   \
  int win = x_dims[3];                  \
  int hin = x_dims[2];                  \
  int chin = x_dims[1];                 \
  int num = x_dims[0];                  \
  int wout = o_dims[3];                 \
  int hout = o_dims[2];                 \
  int chout = o_dims[1];                \
  int kw = w_dims[3];                   \
  int kh = w_dims[2];                   \
  int group = param.groups;             \
  int m = chout / group;                \
  int n = hout * wout;                  \
  int k = chin * kw * kh / group;

#define PREPARE_PARAM                                                         \
  auto& param = this->Param<param_t>();                                       \
  const int input_channel = param.x->dims()[1];                               \
  const int output_channel = param.filter->dims()[0];                         \
  const int groups = param.groups;                                            \
  const int kernel_h = param.filter->dims()[2];                               \
  const int kernel_w = param.filter->dims()[3];                               \
  const int stride_h = param.strides[0];                                      \
  const int stride_w = param.strides[1];                                      \
  auto paddings = *param.paddings;                                            \
  auto dilations = *param.dilations;                                          \
  bool dw_kernel = (input_channel == groups && output_channel == groups);     \
  bool ks_equal = (stride_h == stride_w) && (kernel_h == kernel_w);           \
  bool no_dilation = (dilations[0] == 1) && (dilations[1] == 1);              \
  bool kps_equal = (paddings[0] == paddings[2]) && ks_equal;                  \
  bool pads_equal =                                                           \
      ((paddings[0] == paddings[1]) && (paddings[2] == paddings[3]));         \
  bool flag_dw_3x3 =                                                          \
      (kernel_h == 3) && (kernel_w == 3) && (stride_h == 1 || stride_h == 2); \
  bool flag_dw_5x5 =                                                          \
      (kernel_h == 5) && (kernel_w == 5) && (stride_h == 1 || stride_h == 2);

template <>
void Conv2dCompute<PRECISION(kFloat), PRECISION(kFloat)>::PrepareForRun() {
  PREPARE_PARAM
  //! todo add conv_5x5_depthwise implement
  flag_dw_5x5 = false;
  bool flag_dw = flag_dw_3x3 || flag_dw_5x5;
  if (kernel_w == 1 && stride_w == 1 && paddings[0] == 0 && kps_equal &&
      pads_equal) {
    flag_1x1gemm_ = true;
  } else {
    flag_1x1gemm_ = false;
  }

  //! select conv impl
  if (dw_kernel && kps_equal && no_dilation && flag_dw && (groups & 3) == 0) {
    impl_ = new DepthwiseConv<PRECISION(kFloat), PRECISION(kFloat)>;
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
  auto& ctx = ctx_->As<X86Context>();
  INIT_PARAM
  bool flag_bias = (param.bias != nullptr);
  int group_size_out = m * n;
  int group_size_weights = m * k;
  int group_size_coldata = n * k;
  int channel_in_size = chin * hin * win;
  int channel_out_size = chout * hout * wout;
  auto paddings = *param.paddings;
  auto dilations = *param.dilations;

  auto din = param.x->data<float>();
  auto dout = param.output->mutable_data<float>();
  auto weights = param.filter->data<float>();
  const float* bias_ptr =
      flag_bias ? static_cast<const float*>(param.bias->data<float>())
                : nullptr;
  float* col_data = nullptr;

  if (!flag_1x1gemm_) {
    int col_size = group * group_size_coldata;
    col_data = static_cast<float*>(
        TargetMalloc(TARGET(kX86), col_size * sizeof(float)));
  }
  auto act_param = param.activation_param;
  paddle::lite::x86::math::Blas<lite::TargetType::kX86> matmul(ctx);
  for (int i = 0; i < num; i++) {
    const float* din_batch = din + i * channel_in_size;
    float* dout_batch = dout + i * channel_out_size;
    const float* din_data = din_batch;
    if (!flag_1x1gemm_) {
      lite::x86::math::im2col<float>(din_batch,
                                     chin,
                                     hin,
                                     win,
                                     w_dims[2],
                                     w_dims[3],
                                     paddings[0],
                                     paddings[1],
                                     paddings[2],
                                     paddings[3],
                                     param.strides[0],
                                     param.strides[1],
                                     dilations[0],
                                     dilations[1],
                                     col_data);
      din_data = static_cast<const float*>(col_data);
    }

    for (int g = 0; g < group; g++) {
      const float* col_data_group = din_data + g * group_size_coldata;
      const float* weights_group = weights + g * group_size_weights;
      float* dout_group = dout_batch + g * group_size_out;
      if (n == 1) {
        matmul.GEMV<float>(
            false, m, k, 1.f, weights_group, col_data_group, 0.f, dout_group);
      } else {
        matmul.GEMM<float>(false,
                           false,
                           m,
                           n,
                           k,
                           1.f,
                           weights_group,
                           k,
                           col_data_group,
                           n,
                           0.f,
                           dout_group,
                           n);
      }
    }
    //! bias and activate
    lite::x86::math::fill_bias_act(
        dout_batch, bias_ptr, chout, wout * hout, flag_bias, &act_param);
  }
  if (!flag_1x1gemm_) TargetFree(TARGET(kX86), col_data);
}

template <>
void Conv2dCompute<PRECISION(kInt8), PRECISION(kFloat)>::PrepareForRun() {
  PREPARE_PARAM
  //! todo add conv_5x5_depthwise implement
  flag_dw_5x5 = false;
  bool flag_dw = flag_dw_3x3 || flag_dw_5x5;
  if (kernel_w == 1 && stride_w == 1 && paddings[0] == 0 && kps_equal &&
      pads_equal) {
    flag_1x1gemm_ = true;
  } else {
    flag_1x1gemm_ = false;
  }

  //! select conv impl
  if (dw_kernel && kps_equal && no_dilation && flag_dw) {
    impl_ = new DepthwiseConv<PRECISION(kInt8), PRECISION(kFloat)>;
  } else {
    //! update scale
    w_scale_ = param.weight_scale;
    if (w_scale_.size() != 1 && w_scale_.size() != param.filter->dims()[0]) {
      LOG(FATAL) << "weights scale size must equal to filter size";
      return;
    }
    if (w_scale_.size() == 1) {
      for (int i = 0; i < param.filter->dims()[0] - 1; ++i) {
        w_scale_.push_back(w_scale_[0]);
      }
    }
    float input_scale = param.input_scale;
    for (auto& ws : w_scale_) {
      ws *= input_scale;
    }
  }

  if (impl_) {
    impl_->SetContext(std::move(this->ctx_));
    impl_->SetParam(param);
    impl_->PrepareForRun();
    is_first_epoch_ = false;
  }
}

template <>
void Conv2dCompute<PRECISION(kInt8), PRECISION(kFloat)>::Run() {
  if (impl_) {
    return impl_->Run();
  }
  //! todo add int8 gemm
}

template <>
void Conv2dCompute<PRECISION(kInt8), PRECISION(kInt8)>::PrepareForRun() {
  PREPARE_PARAM
  // todo add conv_5x5_depthwise implement
  flag_dw_5x5 = false;
  bool flag_dw = flag_dw_3x3 || flag_dw_5x5;
  if (kernel_w == 1 && stride_w == 1 && paddings[0] == 0 && kps_equal &&
      pads_equal) {
    flag_1x1gemm_ = true;
  } else {
    flag_1x1gemm_ = false;
  }

  //! select conv impl
  if (dw_kernel && kps_equal && no_dilation && flag_dw) {
    impl_ = new DepthwiseConv<PRECISION(kInt8), PRECISION(kInt8)>;
  } else {
    //! update scale
    w_scale_ = param.weight_scale;
    if (w_scale_.size() != 1 && w_scale_.size() != param.filter->dims()[0]) {
      LOG(FATAL) << "weights scale size must equal to filter size";
      return;
    }
    if (w_scale_.size() == 1) {
      for (int i = 0; i < param.filter->dims()[0] - 1; ++i) {
        w_scale_.push_back(w_scale_[0]);
      }
    }
    float input_scale = param.input_scale;
    float output_scale = param.output_scale;
    for (auto& ws : w_scale_) {
      ws = ws * input_scale / output_scale;
    }
    //!  update bias
    if (param.bias) {
      bias_.Resize(param.bias->dims());
      auto ptr = bias_.mutable_data<float>();
      auto ptr_in = param.bias->data<float>();
      for (int i = 0; i < bias_.numel(); ++i) {
        ptr[i] = ptr_in[i] / param.output_scale;
      }
      flag_trans_bias_ = true;
    }
    //! update relu6 parameter
    if (param.activation_param.active_type ==
        lite_api::ActivationType::kRelu6) {
      param.activation_param.Relu_clipped_coef =
          param.activation_param.Relu_clipped_coef / param.output_scale;
    }
    //! update leakyRelu parameter
    if (param.activation_param.active_type ==
        lite_api::ActivationType::kLeakyRelu) {
      param.activation_param.Leaky_relu_alpha =
          param.activation_param.Leaky_relu_alpha / param.output_scale;
    }
  }

  if (impl_) {
    impl_->SetContext(std::move(this->ctx_));
    impl_->SetParam(param);
    impl_->PrepareForRun();
    is_first_epoch_ = false;
  }
}

template <>
void Conv2dCompute<PRECISION(kInt8), PRECISION(kInt8)>::Run() {
  if (impl_) {
    return impl_->Run();
  }
  //! todo add int8 gemm
}
#undef PREPARE_PARAM
#undef INIT_PARAM
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
