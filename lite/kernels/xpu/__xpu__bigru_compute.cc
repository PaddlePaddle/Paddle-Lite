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

#include "lite/kernels/xpu/__xpu__bigru_compute.h"
#include <vector>
#include "lite/backends/xpu/math.h"
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"
#include "lite/kernels/xpu/mul_compute.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

void XPUBiGRUCompute::PrepareBiasForRun(bool forward) {
  auto& bias_guard_ = forward ? fw_bias_guard_ : bw_bias_guard_;
  auto& param = this->Param<param_t>();
  auto* mul_bias = forward ? param.fw_mul_b : param.bw_mul_b;
  auto* gru_bias = forward ? param.fw_gru_b : param.bw_gru_b;
  if ((mul_bias == nullptr) && (gru_bias == nullptr)) {
    return;
  }
  auto* mul_weight = forward ? param.fw_mul_w : param.bw_mul_w;
  int bias_len = mul_weight->dims()[1];
  std::vector<float> bias_vector(bias_len, 0);
  if (mul_bias) {
    for (int i = 0; i < bias_len; i++) {
      bias_vector[i] += mul_bias->data<float>()[i];
    }
  }
  if (gru_bias) {
    for (int i = 0; i < bias_len; i++) {
      bias_vector[i] += gru_bias->data<float>()[i];
    }
  }
  bias_guard_ = TargetWrapperXPU::MallocScratchPad(bias_len * sizeof(float));
  XPU_CALL(xpu_memcpy(reinterpret_cast<float*>(bias_guard_->addr_),
                      bias_vector.data(),
                      bias_len * sizeof(float),
                      XPUMemcpyKind::XPU_HOST_TO_DEVICE));
}

void XPUBiGRUCompute::PrepareMulWeightForRun(bool forward) {
  auto& weight_abs_max =
      forward ? fw_mul_weight_abs_max_ : bw_mul_weight_abs_max_;
  auto& weight_max_guard =
      forward ? fw_mul_weight_max_guard_ : bw_mul_weight_max_guard_;
  auto& quant_weight_guard =
      forward ? fw_mul_quant_weight_guard_ : bw_mul_quant_weight_guard_;
  auto& param = this->Param<param_t>();
  auto* weight = forward ? param.fw_mul_w : param.bw_mul_w;
  auto weight_ptr = weight->data<float>();
  auto weight_dims = weight->dims();
  auto weight_len = weight->numel();
  // max
  weight_abs_max = paddle::lite::xpu::math::FindMaxAbs(weight_ptr, weight_len);
  std::vector<float> weight_max_vector(4, weight_abs_max);
  weight_max_guard = TargetWrapperXPU::MallocScratchPad(4 * sizeof(float));
  XPU_CALL(xpu_memcpy(reinterpret_cast<float*>(weight_max_guard->addr_),
                      weight_max_vector.data(),
                      4 * sizeof(float),
                      XPUMemcpyKind::XPU_HOST_TO_DEVICE));
  // transpose
  std::vector<float> transpose_weight(weight_len, 0);
  paddle::lite::xpu::math::Transpose(
      weight_ptr, transpose_weight.data(), weight_dims[0], weight_dims[1]);
  // quant
  std::vector<int16_t> quant_weight_cpu(weight_len, 0);
  paddle::lite::xpu::math::ConvertFP32ToInt16(transpose_weight.data(),
                                              quant_weight_cpu.data(),
                                              weight_abs_max,
                                              weight_len);
  quant_weight_guard =
      TargetWrapperXPU::MallocScratchPad(weight_len * sizeof(int16_t));
  XPU_CALL(xpu_memcpy(reinterpret_cast<int16_t*>(quant_weight_guard->addr_),
                      quant_weight_cpu.data(),
                      weight_len * sizeof(int16_t),
                      XPUMemcpyKind::XPU_HOST_TO_DEVICE));
}

void XPUBiGRUCompute::PrepareGRUWeightForRun(bool forward) {
  auto& weight_s1_abs_max_ =
      forward ? fw_gru_weight_s1_abs_max_ : bw_gru_weight_s1_abs_max_;
  auto& weight_s2_abs_max_ =
      forward ? fw_gru_weight_s2_abs_max_ : bw_gru_weight_s2_abs_max_;
  auto& weight_max_guard_ =
      forward ? fw_gru_weight_max_guard_ : bw_gru_weight_max_guard_;
  auto& quant_weight_guard_ =
      forward ? fw_gru_quant_weight_guard_ : bw_gru_quant_weight_guard_;
  //
  auto& param = this->Param<param_t>();
  auto* weight = forward ? param.fw_gru_w : param.bw_gru_w;
  auto weight_ptr = weight->data<float>();
  auto weight_dims = weight->dims();
  int frame_size = weight_dims[0];
  int weight_len = weight->numel();
  // ptr and len
  float* weight_s1_ptr = const_cast<float*>(weight_ptr);
  int weight_s1_len = frame_size * 2 * frame_size;
  float* weight_s2_ptr = weight_s1_ptr + weight_s1_len;
  int weight_s2_len = frame_size * frame_size;
  CHECK_EQ(weight_len, weight_s1_len + weight_s2_len);
  // max
  weight_s1_abs_max_ =
      paddle::lite::xpu::math::FindMaxAbs(weight_s1_ptr, weight_s1_len);
  weight_s2_abs_max_ =
      paddle::lite::xpu::math::FindMaxAbs(weight_s2_ptr, weight_s2_len);
  std::vector<float> weight_max_vector(8);
  for (int i = 0; i < 4; i++) {
    weight_max_vector[i] = weight_s1_abs_max_;
    weight_max_vector[i + 4] = weight_s2_abs_max_;
  }
  weight_max_guard_ = TargetWrapperXPU::MallocScratchPad(8 * sizeof(float));
  XPU_CALL(xpu_memcpy(reinterpret_cast<float*>(weight_max_guard_->addr_),
                      weight_max_vector.data(),
                      8 * sizeof(float),
                      XPUMemcpyKind::XPU_HOST_TO_DEVICE));
// quant
#if 1
  quant_weight_guard_ =
      TargetWrapperXPU::MallocScratchPad(weight_len * sizeof(float));
  XPU_CALL(xpu_memcpy(reinterpret_cast<float*>(quant_weight_guard_->addr_),
                      weight_ptr,
                      weight_len * sizeof(float),
                      XPUMemcpyKind::XPU_HOST_TO_DEVICE));
#else
  quant_weight_guard_ =
      TargetWrapperXPU::MallocScratchPad(weight_len * sizeof(int16_t));
  std::vector<int16_t> quant_weight_cpu(weight_len);
  int16_t* quant_weight_s1_cpu_ptr = quant_weight_cpu.data();
  int16_t* quant_weight_s2_cpu_ptr = quant_weight_s1_cpu_ptr + weight_s1_len;
  paddle::lite::xpu::math::ConvertFP32ToInt16(weight_s1_ptr,
                                              quant_weight_s1_cpu_ptr,
                                              weight_s1_abs_max_,
                                              weight_s1_len);
  paddle::lite::xpu::math::ConvertFP32ToInt16(weight_s2_ptr,
                                              quant_weight_s2_cpu_ptr,
                                              weight_s2_abs_max_,
                                              weight_s2_len);
  XPU_CALL(xpu_memcpy(reinterpret_cast<int16_t*>(quant_weight_guard_->addr_),
                      quant_weight_cpu.data(),
                      weight_len * sizeof(int16_t),
                      XPUMemcpyKind::XPU_HOST_TO_DEVICE));
#endif
}

void XPUBiGRUCompute::PrepareForRun() {
  for (auto forward : {true, false}) {
    PrepareBiasForRun(forward);
    PrepareMulWeightForRun(forward);
    PrepareGRUWeightForRun(forward);
  }
}

void XPUBiGRUCompute::MulRun(bool forward) {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->As<XPUContext>();

  auto& origin_x = *param.input;
  auto& origin_y = forward ? *param.fw_mul_w : *param.bw_mul_w;
  auto x_num_col_dims =
      forward ? param.fw_mul_x_num_col_dims : param.bw_mul_x_num_col_dims;
  auto y_num_col_dims =
      forward ? param.fw_mul_y_num_col_dims : param.bw_mul_y_num_col_dims;
  auto& x_dims = origin_x.dims();
  auto& y_dims = origin_y.dims();
  Tensor x_matrix, y_matrix;
  if (x_dims.size() > 2) {
    x_matrix = ReshapeToMatrix(origin_x, x_num_col_dims);
  } else {
    x_matrix = origin_x;
  }
  if (y_dims.size() > 2) {
    y_matrix = ReshapeToMatrix(origin_y, y_num_col_dims);
  } else {
    y_matrix = origin_y;
  }
  int m = x_matrix.dims()[0];
  int k = x_matrix.dims()[1];
  int n = y_matrix.dims()[1];
  auto& output = forward ? fw_mul_out : bw_mul_out;
  output.Resize({m, n});

  auto& quant_weight_guard =
      forward ? fw_mul_quant_weight_guard_ : bw_mul_quant_weight_guard_;
  auto& weight_abs_max =
      forward ? fw_mul_weight_abs_max_ : bw_mul_weight_abs_max_;
  auto& bias_guard = forward ? fw_bias_guard_ : bw_bias_guard_;

  int r = xdnn::fc_int16(
      ctx.GetRawContext(), /* context */
      false,               /* TransA */
      true,                /* TransB */
      m,
      n,
      k,
      1.0f,                                                        /* alpha */
      x_matrix.data<float>(),                                      /* A */
      reinterpret_cast<const int16_t*>(quant_weight_guard->addr_), /* B */
      weight_abs_max,
      0.0f,                                     /* beta */
      output.mutable_data<float>(TARGET(kXPU)), /* C */
      reinterpret_cast<const float*>(bias_guard->addr_),
      xdnn::Activation_t::LINEAR);
  *(output.mutable_lod()) = origin_x.lod();
  CHECK_EQ(r, 0);
}

void XPUBiGRUCompute::GRURun(bool forward) {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->As<XPUContext>();

  bool origin_mode =
      forward ? param.fw_gru_origin_mode : param.bw_gru_origin_mode;
  bool is_reverse = forward ? false : true;
  auto gate_activation = GetActType(forward ? param.fw_gru_gate_activation
                                            : param.bw_gru_gate_activation);
  auto activation =
      GetActType(forward ? param.fw_gru_activation : param.bw_gru_activation);

  auto* input = forward ? &fw_mul_out : &bw_mul_out;
  const float* input_ptr = input->data<float>();

  const float* hidden_prev_ptr = nullptr;

  const float* weight_ptr = reinterpret_cast<const float*>(
      forward ? fw_gru_quant_weight_guard_->addr_
              : bw_gru_quant_weight_guard_->addr_);
  const float* weight_maxptr =
      reinterpret_cast<const float*>(forward ? fw_gru_weight_max_guard_->addr_
                                             : bw_gru_weight_max_guard_->addr_);

  const float* bias_ptr = nullptr;

  auto* hidden = forward ? param.fw_output : param.bw_output;
  float* hidden_ptr = hidden->mutable_data<float>(TARGET(kXPU));
  const auto& hidden_dims = hidden->dims();
  int frame_size = hidden_dims[1];

  auto& input_lod = input->lod()[0];
  int batch_size = input_lod.size() - 1;
  for (int i = 0; i < batch_size; i++) {
    int cur_seq_len = input_lod[i + 1] - input_lod[i];
    int ret = xdnn::gru_core<float, float, float, int16_t>(ctx.GetRawContext(),
                                                           input_ptr,
                                                           hidden_prev_ptr,
                                                           weight_ptr,
                                                           hidden_ptr,
                                                           1,
                                                           cur_seq_len,
                                                           frame_size,
                                                           nullptr,
                                                           nullptr,
                                                           weight_maxptr,
                                                           nullptr,
                                                           bias_ptr,
                                                           activation,
                                                           gate_activation,
                                                           origin_mode,
                                                           is_reverse);
    CHECK_EQ(ret, 0) << "call xdnn::gru_core failed!";
    input_ptr += cur_seq_len * 3 * frame_size;
    hidden_ptr += cur_seq_len * frame_size;
  }
}

void XPUBiGRUCompute::BiGRURun() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->As<XPUContext>();

  int fw_gru_frame_size = param.fw_output->dims()[1];
  int bw_gru_frame_size = param.bw_output->dims()[1];
  if ((fw_gru_frame_size == bw_gru_frame_size) &&
      (param.fw_gru_origin_mode == param.bw_gru_origin_mode) &&
      (param.fw_gru_gate_activation == param.bw_gru_gate_activation) &&
      (param.fw_gru_activation == param.bw_gru_activation)) {
    bool origin_mode = param.fw_gru_origin_mode;
    auto gate_activation = GetActType(param.fw_gru_gate_activation);
    auto activation = GetActType(param.fw_gru_activation);

    auto* fw_input = &fw_mul_out;
    auto* bw_input = &bw_mul_out;
    const float* fw_input_ptr = fw_input->data<float>();
    const float* bw_input_ptr = bw_input->data<float>();

    const float* fw_weight_ptr =
        reinterpret_cast<const float*>(fw_gru_quant_weight_guard_->addr_);
    const float* bw_weight_ptr =
        reinterpret_cast<const float*>(bw_gru_quant_weight_guard_->addr_);

    const float* fw_weight_maxptr =
        reinterpret_cast<const float*>(fw_gru_weight_max_guard_->addr_);
    const float* bw_weight_maxptr =
        reinterpret_cast<const float*>(bw_gru_weight_max_guard_->addr_);

    const float* fw_bias_ptr = nullptr;
    const float* bw_bias_ptr = nullptr;

    float* fw_hidden_ptr = param.fw_output->mutable_data<float>(TARGET(kXPU));
    float* bw_hidden_ptr = param.bw_output->mutable_data<float>(TARGET(kXPU));

    int frame_size = fw_gru_frame_size;
    auto& input_lod = fw_input->lod()[0];
    int batch_size = input_lod.size() - 1;
    for (int i = 0; i < batch_size; i++) {
      int cur_seq_len = input_lod[i + 1] - input_lod[i];
      int ret =
          xdnn::bigru_core<float, float, float, int16_t>(ctx.GetRawContext(),
                                                         fw_input_ptr,
                                                         bw_input_ptr,
                                                         nullptr,
                                                         nullptr,
                                                         fw_weight_ptr,
                                                         bw_weight_ptr,
                                                         fw_hidden_ptr,
                                                         bw_hidden_ptr,
                                                         1,
                                                         cur_seq_len,
                                                         frame_size,
                                                         nullptr,
                                                         nullptr,
                                                         nullptr,
                                                         nullptr,
                                                         fw_weight_maxptr,
                                                         bw_weight_maxptr,
                                                         nullptr,
                                                         nullptr,
                                                         fw_bias_ptr,
                                                         bw_bias_ptr,
                                                         activation,
                                                         gate_activation,
                                                         origin_mode);
      CHECK_EQ(ret, 0) << "call xdnn::bigru_core failed!";
      fw_input_ptr += cur_seq_len * 3 * frame_size;
      bw_input_ptr += cur_seq_len * 3 * frame_size;
      fw_hidden_ptr += cur_seq_len * frame_size;
      bw_hidden_ptr += cur_seq_len * frame_size;
    }
  } else {
    // FW_GRU
    GRURun(true);
    // BW_GRU
    GRURun(false);
  }
}

void XPUBiGRUCompute::Run() {
  // FW_MUL
  MulRun(true);
  // BW_MUL
  MulRun(false);
  // BiGRU
  BiGRURun();
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(__xpu__bigru,
                     kXPU,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::xpu::XPUBiGRUCompute,
                     def)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("ForwardMulWeight", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindInput("ForwardMulBias", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindInput("ForwardGRUWeight", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindInput("ForwardGRUBias", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindInput("BackwardMulWeight", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindInput("BackwardMulBias", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindInput("BackwardGRUWeight", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindInput("BackwardGRUBias", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("ForwardOutput", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("BackwardOutput", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();
