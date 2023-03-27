// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/kernels/arm/fused_attention_compute.h"
#include <string>
#include <vector>
#include "lite/backends/arm/math/gemm_prepacked_int8.h"
#include "lite/backends/arm/math/gemm_s8.h"
#include "lite/backends/arm/math/norm.h"
#include "lite/backends/arm/math/softmax.h"
#include "lite/backends/arm/math/type_trans.h"
#include "lite/core/op_registry.h"
#include "lite/core/parallel_defines.h"
#include "lite/kernels/arm/layer_norm_compute.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

void TransposeCompute_1to3(int8_t* input,
                           int8_t* v0,
                           int8_t* v1,
                           int8_t* v2,
                           int input_c,
                           int intput_h,
                           int intput_w,
                           int output_w) {
  int src_index = 0;
  int dst_index = 0;
  int w_block = intput_w / 3 / output_w;
  int out_hw_stride = output_w * intput_h;
  int in_hw_stride = intput_h * intput_w;
  for (int c = 0; c < input_c; c++) {
    for (int h = 0; h < intput_h; h++) {
      for (int w = 0; w < w_block; w++) {
        memcpy(v0 + w * out_hw_stride, input, output_w * sizeof(int8_t));
        input += output_w;
      }
      for (int w = 0; w < w_block; w++) {
        memcpy(v1 + w * out_hw_stride, input, output_w * sizeof(int8_t));
        input += output_w;
      }
      for (int w = 0; w < w_block; w++) {
        memcpy(v2 + w * out_hw_stride, input, output_w * sizeof(int8_t));
        input += output_w;
      }
      v0 += output_w;
      v1 += output_w;
      v2 += output_w;
    }
  }
}

template <>
void FusedAttentionCompute<PRECISION(kInt8)>::PrepareForRun() {
  ReInitWhenNeeded();
  auto& param = this->template Param<param_t>();
  // prepack weight
  auto& ctx = this->ctx_->template As<ARMContext>();
  auto* w_data = param.fc_w->template data<int8_t>();
  size_t llc_size = ctx.llc_size() / 4;
  if (ctx.has_dot()) {
#ifdef WITH_ARM_DOTPROD
    int x_block = (llc_size - (lite::arm::math::MBLOCK_INT8_DOT * fc_k_)) /
                  (sizeof(int8_t) * (fc_k_ + lite::arm::math::MBLOCK_INT8_DOT));
    x_block /= lite::arm::math::NBLOCK_INT8_DOT;
    x_block *= lite::arm::math::NBLOCK_INT8_DOT;

    int x_num = (fc_n_ + (x_block - 1)) / x_block;
    x_block = (fc_n_ + x_num - 1) / x_num;
    x_block = (x_block + lite::arm::math::NBLOCK_INT8_DOT - 1) /
              lite::arm::math::NBLOCK_INT8_DOT;
    x_block *= lite::arm::math::NBLOCK_INT8_DOT;
    x_block = x_block < lite::arm::math::NBLOCK_INT8_DOT
                  ? lite::arm::math::NBLOCK_INT8_DOT
                  : x_block;
    int kup = ROUNDUP(fc_k_, lite::arm::math::KBLOCK_INT8);
    DDim prepack_w_dim =
        DDim(std::vector<int64_t>{kup, ROUNDUP(fc_n_, x_block)});
    Tensor prepack_w;
    prepack_w.Resize(prepack_w_dim);
    int8_t* prepack_w_data = prepack_w.mutable_data<int8_t>();

    for (unsigned int x0 = 0; x0 < fc_n_; x0 += x_block) {
      unsigned int xmax = x0 + x_block;
      xmax = (xmax > fc_n_) ? fc_n_ : xmax;
#ifdef __aarch64__
      lite::arm::math::packb_sdot_int8_n12_n8_n4(
          prepack_w_data + x0 * kup, w_data, fc_n_, 0, fc_k_, x0, xmax);
#else
      lite::arm::math::packb_dot_int8(
          prepack_w_data + x0 * kup, w_data, fc_n_, 0, fc_k_, x0, xmax);
#endif
    }
    DDim w_origin_dim = param.fc_w->dims();
    param.fc_w->CopyDataFrom(prepack_w);
    param.fc_w->Resize(w_origin_dim);
#endif
  } else {
    const int kup = ROUNDUP(fc_k_, lite::arm::math::KBLOCK_INT8);
    int x_block =
        llc_size / (sizeof(int8_t) * (kup + lite::arm::math::MBLOCK_INT8_OTH));
    x_block /= lite::arm::math::NBLOCK_INT8_OTH;
    x_block *= lite::arm::math::NBLOCK_INT8_OTH;
    int x_num = (fc_n_ + (x_block - 1)) / x_block;
    x_block = (fc_n_ + x_num - 1) / x_num;
    x_block = (x_block + lite::arm::math::NBLOCK_INT8_OTH - 1) /
              lite::arm::math::NBLOCK_INT8_OTH;
    x_block *= lite::arm::math::NBLOCK_INT8_OTH;

    DDim prepack_w_dim =
        DDim(std::vector<int64_t>{kup, ROUNDUP(fc_n_, x_block)});
    Tensor prepack_w;
    prepack_w.Resize(prepack_w_dim);
    int8_t* prepack_w_data = prepack_w.mutable_data<int8_t>();

    auto* zerobuf = static_cast<int8_t*>(malloc(x_block * (sizeof(int8_t))));
    memset(zerobuf, 0, x_block * sizeof(int8_t));
    for (unsigned int x0 = 0; x0 < fc_n_; x0 += x_block) {
      unsigned int xmax = x0 + x_block;
      if (xmax >= fc_n_) {
        xmax = fc_n_;
      }
      lite::arm::math::packb_int8(prepack_w_data + x0 * kup,
                                  w_data,
                                  fc_n_,
                                  0,
                                  fc_k_,
                                  x0,
                                  xmax,
                                  zerobuf);
    }
    DDim w_origin_dim = param.fc_w->dims();
    param.fc_w->CopyDataFrom(prepack_w);
    param.fc_w->Resize(w_origin_dim);
    free(zerobuf);
  }
}

template <PrecisionType PType>
void FusedAttentionCompute<PType>::ReInitWhenNeeded() {
  auto& param = this->template Param<param_t>();
  auto input_dims = param.input->dims();
  if (last_shape_ == input_dims) {
    return;
  }
  last_shape_ = input_dims;
  // fc
  act_param_.has_active = false;
  if (param.activation_type == "relu") {
    act_param_.has_active = true;
    act_param_.active_type = lite_api::ActivationType::kRelu;
  } else if (param.activation_type == "relu6") {
    act_param_.has_active = true;
    act_param_.active_type = lite_api::ActivationType::kRelu6;
    act_param_.Relu_clipped_coef = param.alpha;
  }
  auto w_dims = param.fc_w->dims();
  int in_num_col_dims = param.in_num_col_dims;
  std::string op_type = param.op_type;
  if (op_type == "matmul" || op_type == "matmul_v2") {
    in_num_col_dims = input_dims.size() - 1;
  }
  fc_m_ = input_dims.Slice(0, in_num_col_dims).production();
  fc_k_ = input_dims.Slice(in_num_col_dims, input_dims.size()).production();
  CHECK_EQ(fc_k_, w_dims[0]);
  fc_n_ = w_dims[1];
  fc_dims_ = DDim(std::vector<int64_t>{input_dims[0], fc_m_, fc_n_});

  // reshape
  reshape_shape_.push_back(input_dims[0]);
  reshape_shape_.push_back(param.reshape_shape[2]);
  reshape_shape_.push_back(input_dims[1]);
  reshape_shape_.push_back(param.reshape_shape[3]);

  // transpose
  transpose_out_dim_ = DDim(std::vector<int64_t>{
      input_dims[0], reshape_shape_[1], fc_m_, reshape_shape_[3]});
  // fc1
  fc1_m_ = transpose_out_dim_[2];
  fc1_n_ = transpose_out_dim_[2];
  fc1_k_ = transpose_out_dim_[3];
  fc1_out_dim_ = DDim(std::vector<int64_t>{
      input_dims[0], transpose_out_dim_[1], fc1_m_, fc1_n_});

  // softmax
  softmax_out_dim_ = fc1_out_dim_;

  // calib1
  calib1_dims_ = softmax_out_dim_;

  // fc2
  fc2_m_ = softmax_out_dim_[2];
  fc2_n_ = transpose_out_dim_[3];
  fc2_k_ = softmax_out_dim_[3];

  if (param.enable_int8) {
    for (int i = 0; i < fc2_n_; i++) {
      fc2_scale_.push_back(param.fc2_scale.data()[0]);
    }
    for (int i = 0; i < fc1_n_; i++) {
      fc1_scale_.push_back(param.fc1_scale.data()[0]);
    }
  }
}

template <>
void FusedAttentionCompute<PRECISION(kInt8)>::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->template As<ARMContext>();
  auto* input_data = param.input->data<int8_t>();
  auto* residual_data = param.residual->data<float>();
  auto* o_data = param.output->mutable_data<float>();
  auto input_dims = param.input->dims();
  auto out_dims = param.output->dims();

  // fc + dequant_scale, bias, quant_scale
  Tensor fc_t;
  fc_t.Resize(fc_dims_);
  fc_t.mutable_data<int8_t>();
  auto* fc_out = const_cast<int8_t*>(fc_t.data<int8_t>());
  auto* w_data = param.fc_w->data<int8_t>();
  auto* b_data = param.fc_bias ? param.fc_bias->data<float>() : nullptr;
  lite::arm::math::gemm_s8<int8_t>(false,
                                   false,
                                   fc_m_,
                                   fc_n_,
                                   fc_k_,
                                   input_data,
                                   w_data,
                                   fc_out,
                                   b_data,
                                   true,
                                   lite::arm::math::GemmNBias,
                                   param.fc0_scale.data(),
                                   act_param_,
                                   &ctx,
                                   true);
  // transpose2 fuse reshape2
  DDim trans_dims = DDim(std::vector<int64_t>{
      input_dims[0], reshape_shape_[1], fc_m_, reshape_shape_[3] * 3});
  Tensor trans_t;
  trans_t.Resize(trans_dims);
  trans_t.mutable_data<int8_t>();
  auto* trans_out = const_cast<int8_t*>(trans_t.data<int8_t>());
  int stride = fc_m_ * fc_n_ / 3;
  auto* v0 = trans_out;
  auto* v1 = v0 + stride;
  auto* v2 = v1 + stride;
  TransposeCompute_1to3(
      fc_out, v0, v1, v2, input_dims[0], fc_m_, fc_n_, transpose_out_dim_[3]);
  // fc -> out fp32
  Tensor fc1_t;
  fc1_t.Resize(fc1_out_dim_);
  fc1_t.mutable_data<float>();
  auto* fc1_out = const_cast<float*>(fc1_t.data<float>());
  int x_inner = fc1_m_ * fc1_k_;
  int y_inner = fc1_k_ * fc1_n_;
  int out_inner = fc1_m_ * fc1_n_;
  auto* fc1_b_data = param.residual->data<float>();
  for (size_t i = 0; i < transpose_out_dim_[1]; ++i) {
    lite::arm::math::gemm_s8(false,
                             true,
                             fc1_m_,
                             fc1_n_,
                             fc1_k_,
                             v0 + i * x_inner,
                             v1 + i * y_inner,
                             fc1_out + i * out_inner,
                             fc1_b_data,
                             true,
                             lite::arm::math::GemmNBias,
                             fc1_scale_.data(),
                             act_param_,
                             &ctx,
                             false);
  }

  // softmax
  Tensor softmax_t;
  softmax_t.Resize(softmax_out_dim_);
  softmax_t.mutable_data<float>();
  auto* softmax_out = const_cast<float*>(softmax_t.data<float>());
  int axis = param.softmax_axis;
  if (axis < 0) {
    axis += fc1_out_dim_.size();
  }
  int outer_num = fc1_out_dim_.Slice(0, axis).production();
  int inner_num =
      fc1_out_dim_.Slice(axis + 1, fc1_out_dim_.size()).production();
  int axis_size = fc1_out_dim_[axis];

  if (axis_size > 4) {
    lite::arm::math::softmax_inner1_large_axis(
        fc1_out, softmax_out, outer_num, axis_size);
  } else {
    lite::arm::math::softmax_inner1_small_axis(
        fc1_out, softmax_out, outer_num, axis_size);
  }
  // calib fp32 -> int8
  Tensor calib1_t;
  calib1_t.Resize(calib1_dims_);
  calib1_t.mutable_data<int8_t>();
  auto* calib1_out = const_cast<int8_t*>(calib1_t.data<int8_t>());
  lite::arm::math::fp32_to_int8(softmax_out,
                                calib1_out,
                                param.calib1_scale.data(),
                                1,
                                1,
                                softmax_out_dim_.production());

  int fc2_x_inner = fc2_m_ * fc2_k_;
  int fc2_y_inner = fc2_k_ * fc2_n_;
  int fc2_out_inner = fc2_m_ * fc2_n_;
  for (size_t i = 0; i < out_dims[1]; ++i) {
    lite::arm::math::gemm_s8<float>(false,
                                    false,
                                    fc2_m_,
                                    fc2_n_,
                                    fc2_k_,
                                    calib1_out + i * fc2_x_inner,
                                    v2 + i * fc2_y_inner,
                                    o_data + i * fc2_out_inner,
                                    nullptr,
                                    false,
                                    lite::arm::math::GemmNBias,
                                    fc2_scale_.data(),
                                    act_param_,
                                    &ctx,
                                    false);
  }
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
typedef paddle::lite::kernels::arm::FusedAttentionCompute<PRECISION(kInt8)>
    FusedAttentionCompute_Int8;
REGISTER_LITE_KERNEL(
    fused_attention, kARM, kInt8, kNCHW, FusedAttentionCompute_Int8, def)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt8))})
    .BindInput("Residual",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFloat))})
    .BindInput("W", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt8))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFloat))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFloat))})
    .Finalize();
