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

#include "lite/backends/cuda/math/cudnn_conv.h"
#include "lite/backends/cuda/math/activation.h"
#include "lite/backends/cuda/math/scale.h"
#include "lite/backends/cuda/math/type_trans.h"

namespace paddle {
namespace lite {
namespace cuda {
namespace math {

template <>
bool CudnnConv2D<PRECISION(kFloat)>::create(const operators::ConvParam& param,
                                            Context<TARGET(kCUDA)>* ctx) {
  auto x_dims = param.x->dims();
  auto w_dims = param.filter->dims();
  auto o_dims = param.output->dims();
  int batch = x_dims[0];

  int iw = x_dims[3];  // nchw
  int ih = x_dims[2];
  int ic = x_dims[1];
  int ow = o_dims[3];
  int oh = o_dims[2];
  int oc = o_dims[1];
  int kw = w_dims[3];
  int kh = w_dims[2];
  int sw = param.strides[1];
  int sh = param.strides[0];
  int pw = param.paddings[1];
  int ph = param.paddings[0];
  int dw = param.dilations[1];
  int dh = param.dilations[0];

  CHECK(ic % param.groups == 0)
      << "The conv input channel shoud be divide group number.";

  CUDNN_CHECK(cudnnSetTensor4dDescriptor(this->input_desc_,
                                         CUDNN_TENSOR_NCHW,
                                         CUDNN_DATA_FLOAT,
                                         batch,
                                         ic,
                                         ih,
                                         iw));
  CUDNN_CHECK(cudnnSetFilter4dDescriptor(this->filter_desc_,
                                         CUDNN_DATA_FLOAT,
                                         CUDNN_TENSOR_NCHW,
                                         oc,
                                         ic / param.groups,
                                         kh,
                                         kw));
  CUDNN_CHECK(cudnnSetConvolution2dDescriptor(this->conv_desc_,
                                              ph,
                                              pw,
                                              sh,
                                              sw,
                                              dh,
                                              dw,
                                              CUDNN_CROSS_CORRELATION,
                                              CUDNN_DATA_FLOAT));
  CUDNN_CHECK(cudnnSetConvolutionGroupCount(this->conv_desc_, param.groups));
  CUDNN_CHECK(cudnnSetTensor4dDescriptor(this->output_desc_,
                                         CUDNN_TENSOR_NCHW,
                                         CUDNN_DATA_FLOAT,
                                         batch,
                                         oc,
                                         oh,
                                         ow));

  if (param.activation_param.has_active && with_relu_act_) {
    CUDNN_CHECK(cudnnSetActivationDescriptor(
        this->act_desc_, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN, 0.0));
  }

  if (ic == param.groups && ic == oc && ic != 1) {
    this->fwd_algo_ = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
  } else {
    CUDNN_CHECK(
        cudnnGetConvolutionForwardAlgorithm(this->handle_,
                                            this->input_desc_,
                                            this->filter_desc_,
                                            this->conv_desc_,
                                            this->output_desc_,
                                            this->preference_,
                                            this->workspace_limit_bytes_,
                                            &this->fwd_algo_));
  }
  CUDNN_CHECK(
      cudnnGetConvolutionForwardWorkspaceSize(this->handle_,
                                              this->input_desc_,
                                              this->filter_desc_,
                                              this->conv_desc_,
                                              this->output_desc_,
                                              this->fwd_algo_,
                                              &this->workspace_fwd_sizes_));
  if (this->workspace_fwd_sizes_ > this->workspace_size_inbytes_) {
    this->workspace_size_inbytes_ = this->workspace_fwd_sizes_;
    if (this->workspace_data_ != NULL) {
      cudaFree(this->workspace_data_);
    }
    cudaMalloc(&this->workspace_data_, this->workspace_size_inbytes_);
    this->workspace_ = reinterpret_cast<char*>(this->workspace_data_);
  }
  if (param.bias) {
    int dim_bias[] = {1, oc, 1, 1};
    int stride_bias[] = {oc, 1, 1, 1};
    cudnnSetTensorNdDescriptor(
        this->bias_desc_, CUDNN_DATA_FLOAT, 4, dim_bias, stride_bias);
  }
  return true;
}

template <>
bool CudnnConv2D<PRECISION(kFloat)>::init(const operators::ConvParam& param,
                                          Context<TARGET(kCUDA)>* ctx) {
  this->workspace_size_inbytes_ = 0;
  this->workspace_data_ = NULL;
  this->workspace_fwd_sizes_ = 0;

  this->stream_ = ctx->exec_stream();
  CUDNN_CHECK(cudnnCreate(&this->handle_));
  CUDNN_CHECK(cudnnSetStream(this->handle_, this->stream_));

  this->workspace_ = NULL;

  cudnnCreateTensorDescriptor(&this->input_desc_);
  cudnnCreateTensorDescriptor(&this->output_desc_);
  cudnnCreateFilterDescriptor(&this->filter_desc_);
  cudnnCreateConvolutionDescriptor(&this->conv_desc_);
  cudnnCreateTensorDescriptor(&this->bias_desc_);

  if (param.activation_param.has_active) {
    if (param.activation_param.active_type == lite_api::ActivationType::kRelu) {
      cudnnCreateActivationDescriptor(&this->act_desc_);
    } else {
      this->with_relu_act_ = false;
    }
  }
  return create(param, ctx);
}

template <>
bool CudnnConv2D<PRECISION(kFloat)>::run(const operators::ConvParam& param) {
  const auto* i_data = param.x->data<float>();
  const auto* w_data = param.filter->data<float>();
  const auto* b_data = param.bias ? param.bias->data<float>() : nullptr;
  auto* o_data = param.output->mutable_data<float>(TARGET(kCUDA));

  if (param.activation_param.has_active && with_relu_act_) {
    if (b_data) {
      float alpha = 1.0f;
      float beta = 0.0f;
      CUDNN_CHECK(cudnnConvolutionBiasActivationForward(handle_,
                                                        &alpha,
                                                        input_desc_,
                                                        i_data,
                                                        filter_desc_,
                                                        w_data,
                                                        conv_desc_,
                                                        fwd_algo_,
                                                        workspace_,
                                                        workspace_fwd_sizes_,
                                                        &beta,
                                                        output_desc_,
                                                        o_data,
                                                        bias_desc_,
                                                        b_data,
                                                        act_desc_,
                                                        output_desc_,
                                                        o_data));
    } else {
      float alpha = 1.0f;
      float beta = 0.0f;
      CUDNN_CHECK(cudnnConvolutionForward(handle_,
                                          &alpha,
                                          input_desc_,
                                          i_data,
                                          filter_desc_,
                                          w_data,
                                          conv_desc_,
                                          fwd_algo_,
                                          workspace_,
                                          workspace_fwd_sizes_,
                                          &beta,
                                          output_desc_,
                                          o_data));

      CUDNN_CHECK(cudnnActivationForward(handle_,
                                         act_desc_,
                                         &alpha,
                                         output_desc_,
                                         o_data,
                                         &beta,
                                         output_desc_,
                                         o_data));
    }
  } else {
    float alpha = 1.0f;
    float beta = 0.0f;
    CUDNN_CHECK(cudnnConvolutionForward(handle_,
                                        &alpha,
                                        input_desc_,
                                        i_data,
                                        filter_desc_,
                                        w_data,
                                        conv_desc_,
                                        fwd_algo_,
                                        workspace_,
                                        workspace_fwd_sizes_,
                                        &beta,
                                        output_desc_,
                                        o_data));
    if (b_data) {
      CUDNN_CHECK(cudnnAddTensor(
          handle_, &alpha, bias_desc_, b_data, &alpha, output_desc_, o_data));
    }
  }

  if (!with_relu_act_) {
    CHECK(param.activation_param.active_type ==
          lite_api::ActivationType::kLeakyRelu)
        << "Only support leaky relu now.";
    auto out_dims = param.output->dims();
    int n = out_dims[0], c = out_dims[1], h = out_dims[2], w = out_dims[3];
    int num = n * h * w * c;
    float alpha = param.activation_param.Leaky_relu_alpha;

    relu(num, o_data, o_data, alpha, this->stream_);
  }
  return true;
}

template <PrecisionType Ptype_out>
bool CudnnConv2DInt8<Ptype_out>::create(const operators::ConvParam& param,
                                        Context<TARGET(kCUDA)>* ctx) {
  auto x_dims = param.x->dims();
  auto w_dims = param.filter->dims();
  auto o_dims = param.output->dims();

  int batch = x_dims[0];

  int iw = x_dims[2];  // nchw
  int ih = x_dims[1];
  int ic = x_dims[3];
  int ow = o_dims[2];
  int oh = o_dims[1];
  int oc = o_dims[3];

  int kw = w_dims[2];
  int kh = w_dims[1];

  int sw = param.strides[1];
  int sh = param.strides[0];
  int pw = param.paddings[1];
  int ph = param.paddings[0];
  int dw = param.dilations[1];
  int dh = param.dilations[0];

  std::vector<float> weight_scale = param.weight_scale;
  float input_scale = param.input_scale;
  float output_scale = param.output_scale;
  CHECK(weight_scale.size() == oc)
      << "the num of the weight_scale should be equals to the output channel.";
  if (Ptype_out == PRECISION(kInt8)) {
    this->temp_tensor_.Resize(o_dims);
    this->temp_tensor_.template mutable_data<float>(TARGET(kCUDA));
    for (int i = 0; i < weight_scale.size(); i++) {
      weight_scale[i] = (weight_scale[i] * input_scale) / output_scale;
    }
  } else {
    for (int i = 0; i < weight_scale.size(); i++) {
      weight_scale[i] = (weight_scale[i] * input_scale);
    }
  }
  this->scale_.Resize({oc});
  auto* scale_data = this->scale_.template mutable_data<float>(TARGET(kCUDA));
  this->scale_.template Assign<float, lite::DDim, TARGET(kCUDA)>(
      weight_scale.data(), this->scale_.dims());

  CHECK(ic % param.groups == 0)
      << "The conv input channel shoud be divide group number.";
  CUDNN_CHECK(cudnnSetTensor4dDescriptor(this->input_desc_,
                                         CUDNN_TENSOR_NHWC,
                                         CUDNN_DATA_INT8,
                                         batch,
                                         ic,
                                         ih,
                                         iw));
  CUDNN_CHECK(cudnnSetFilter4dDescriptor(this->filter_desc_,
                                         CUDNN_DATA_INT8,
                                         CUDNN_TENSOR_NHWC,
                                         oc,
                                         ic / param.groups,
                                         kh,
                                         kw));
  CUDNN_CHECK(cudnnSetConvolution2dDescriptor(this->conv_desc_,
                                              ph,
                                              pw,
                                              sh,
                                              sw,
                                              dh,
                                              dw,
                                              CUDNN_CROSS_CORRELATION,
                                              CUDNN_DATA_INT32));

  CUDNN_CHECK(cudnnSetTensor4dDescriptor(this->output_desc_,
                                         CUDNN_TENSOR_NHWC,
                                         CUDNN_DATA_FLOAT,
                                         batch,
                                         oc,
                                         oh,
                                         ow));

  this->fwd_algo_ = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
  CUDNN_CHECK(
      cudnnGetConvolutionForwardWorkspaceSize(this->handle_,
                                              this->input_desc_,
                                              this->filter_desc_,
                                              this->conv_desc_,
                                              this->output_desc_,
                                              this->fwd_algo_,
                                              &(this->workspace_fwd_sizes_)));

  if (this->workspace_fwd_sizes_ > this->workspace_size_inbytes_) {
    this->workspace_size_inbytes_ = this->workspace_fwd_sizes_;
    if (this->workspace_data_ != NULL) {
      cudaFree(this->workspace_data_);
    }
    cudaMalloc(&this->workspace_data_, this->workspace_size_inbytes_);
    this->workspace_ = reinterpret_cast<char*>(this->workspace_data_);
  }

  return true;
}

template <PrecisionType Ptype_out>
bool CudnnConv2DInt8<Ptype_out>::init(const operators::ConvParam& param,
                                      Context<TARGET(kCUDA)>* ctx) {
  this->workspace_size_inbytes_ = 0;  // 64Mb
  this->workspace_data_ = NULL;
  this->workspace_fwd_sizes_ = 0;

  this->stream_ = ctx->exec_stream();
  CUDNN_CHECK(cudnnCreate(&this->handle_));
  CUDNN_CHECK(cudnnSetStream(this->handle_, this->stream_));

  this->workspace_ = NULL;

  cudnnCreateTensorDescriptor(&this->input_desc_);
  cudnnCreateTensorDescriptor(&this->output_desc_);
  cudnnCreateFilterDescriptor(&this->filter_desc_);
  cudnnCreateConvolutionDescriptor(&this->conv_desc_);
  cudnnCreateTensorDescriptor(&this->bias_desc_);

  if (param.activation_param.has_active) {
    if (!(param.activation_param.active_type ==
          lite_api::ActivationType::kRelu)) {
      this->with_relu_act_ = false;
    }
  }
  return create(param, ctx);
}

template <PrecisionType Ptype_out>
bool CudnnConv2DInt8<Ptype_out>::run(const operators::ConvParam& param) {
  const auto* i_data = param.x->data<int8_t>();
  const auto* w_data = param.filter->data<int8_t>();
  const auto* b_data = param.bias ? param.bias->data<float>() : nullptr;
  float* temp_out;
  float* scale = this->scale_.template mutable_data<float>(TARGET(kCUDA));
  if (Ptype_out == PRECISION(kInt8)) {
    temp_out = this->temp_tensor_.template mutable_data<float>(TARGET(kCUDA));
  } else {
    temp_out = param.output->mutable_data<float>(TARGET(kCUDA));
  }

  float alpha = 1.0f;
  float beta = 0.0f;
  CUDNN_CHECK(cudnnConvolutionForward(this->handle_,
                                      &alpha,
                                      this->input_desc_,
                                      i_data,
                                      this->filter_desc_,
                                      w_data,
                                      this->conv_desc_,
                                      this->fwd_algo_,
                                      this->workspace_,
                                      this->workspace_fwd_sizes_,
                                      &beta,
                                      this->output_desc_,
                                      temp_out));

  auto out_dims = param.output->dims();
  int n = out_dims[0], h = out_dims[1], w = out_dims[2], c = out_dims[3];
  int num = n * h * w * c / 4;

  if (!param.activation_param.has_active && !b_data) {
    if (Ptype_out == PRECISION(kInt8)) {
      auto* out = param.output->mutable_data<int8_t>(TARGET(kCUDA));
      fp32_to_int8_nhwc4(num,
                         static_cast<const void*>(temp_out),
                         static_cast<void*>(out),
                         static_cast<const void*>(scale),
                         n,
                         c / 4,
                         h,
                         w,
                         this->stream_);
    } else {
      fp32_scale_nhwc4(num,
                       static_cast<const void*>(temp_out),
                       static_cast<void*>(temp_out),
                       static_cast<const void*>(scale),
                       n,
                       c / 4,
                       h,
                       w,
                       this->stream_);
    }
    return true;
  }

  if (b_data) {
    if (param.activation_param.has_active) {
      float alpha = 0.0;
      if (!this->with_relu_act_)
        alpha = param.activation_param.Leaky_relu_alpha;
      if (Ptype_out == PRECISION(kInt8)) {
        auto* out = param.output->mutable_data<int8_t>(TARGET(kCUDA));
        bias_relu_int8_nhwc4<int8_t>(num,
                                     static_cast<const void*>(temp_out),
                                     static_cast<const void*>(b_data),
                                     static_cast<void*>(out),
                                     n,
                                     c / 4,
                                     h,
                                     w,
                                     static_cast<const void*>(scale),
                                     alpha,
                                     this->stream_);
      } else {
        bias_relu_int8_nhwc4<float>(num,
                                    static_cast<const void*>(temp_out),
                                    static_cast<const void*>(b_data),
                                    static_cast<void*>(temp_out),
                                    n,
                                    c / 4,
                                    h,
                                    w,
                                    static_cast<const void*>(scale),
                                    alpha,
                                    this->stream_);
      }
      return true;
    }
  }

  CHECK(false)
      << "Conv Int8 support Conv, Conv + bias + relu, Conv + bias + leaky_relu";
}

template class CudnnConv2DInt8<PRECISION(kInt8)>;
template class CudnnConv2DInt8<PRECISION(kFloat)>;

}  // namespace math
}  // namespace cuda
}  // namespace lite
}  // namespace paddle
