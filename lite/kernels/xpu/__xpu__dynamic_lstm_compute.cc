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

#include "lite/kernels/xpu/__xpu__dynamic_lstm_compute.h"
#include <algorithm>
#include <vector>
#include "lite/backends/xpu/math.h"
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

void XPUDynamicLstmCompute::PrepareForRun() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();
  int max_ptr_size = ctx.GetRawContext()->max_ptr_size();

  // transpose from weight_0[xdim, 4 * hdim] to transpose_weight_0[4 * hdim,
  // xdim]
  int weight_0_size = param.weight_0->numel();
  auto& weight_0_dims = param.weight_0->dims();
  std::vector<float> cpu_transpose_weight_0(weight_0_size);
  paddle::lite::xpu::math::Transpose(param.weight_0->template data<float>(),
                                     cpu_transpose_weight_0.data(),
                                     weight_0_dims[0],
                                     weight_0_dims[1]);

  // change weight_0 from [w_gx, w_ix, w_fx, w_ox] to [w_ix, w_fx, w_gx, w_ox]
  transpose_weight_0_ =
      TargetWrapperXPU::MallocScratchPad(weight_0_size * sizeof(float));
  float* transpose_weight_0_addr =
      reinterpret_cast<float*>(transpose_weight_0_->addr_);
  XPU_CALL(xpu_memcpy(transpose_weight_0_addr,
                      cpu_transpose_weight_0.data() + weight_0_size / 4,
                      weight_0_size / 4 * sizeof(float),
                      XPUMemcpyKind::XPU_HOST_TO_DEVICE));
  XPU_CALL(xpu_memcpy(transpose_weight_0_addr + weight_0_size / 4,
                      cpu_transpose_weight_0.data() + weight_0_size / 2,
                      weight_0_size / 4 * sizeof(float),
                      XPUMemcpyKind::XPU_HOST_TO_DEVICE));
  XPU_CALL(xpu_memcpy(transpose_weight_0_addr + weight_0_size / 2,
                      cpu_transpose_weight_0.data(),
                      weight_0_size / 2 * sizeof(float),
                      XPUMemcpyKind::XPU_HOST_TO_DEVICE));
  XPU_CALL(xpu_memcpy(transpose_weight_0_addr + weight_0_size / 4 * 3,
                      cpu_transpose_weight_0.data() + weight_0_size / 4 * 3,
                      weight_0_size / 4 * sizeof(float),
                      XPUMemcpyKind::XPU_HOST_TO_DEVICE));

  // transpose from weight_1[hdim, 4 * hdim] to transpose_weight_1[4 * hdim,
  // hdim]
  int weight_1_size = param.weight_1->numel();
  auto& weight_1_dims = param.weight_1->dims();
  std::vector<float> cpu_transpose_weight_1(weight_1_size);
  paddle::lite::xpu::math::Transpose(param.weight_1->template data<float>(),
                                     cpu_transpose_weight_1.data(),
                                     weight_1_dims[0],
                                     weight_1_dims[1]);
  // change weight_1 from [w_gh, w_ih, w_fh, w_oh] to [w_ih, w_fh, w_gh, w_oh]
  transpose_weight_1_ =
      TargetWrapperXPU::MallocScratchPad(weight_1_size * sizeof(float));
  float* transpose_weight_1_addr =
      reinterpret_cast<float*>(transpose_weight_1_->addr_);
  XPU_CALL(xpu_memcpy(transpose_weight_1_addr,
                      cpu_transpose_weight_1.data() + weight_1_size / 4,
                      weight_1_size / 4 * sizeof(float),
                      XPUMemcpyKind::XPU_HOST_TO_DEVICE));
  XPU_CALL(xpu_memcpy(transpose_weight_1_addr + weight_1_size / 4,
                      cpu_transpose_weight_1.data() + weight_1_size / 2,
                      weight_1_size / 4 * sizeof(float),
                      XPUMemcpyKind::XPU_HOST_TO_DEVICE));
  XPU_CALL(xpu_memcpy(transpose_weight_1_addr + weight_1_size / 2,
                      cpu_transpose_weight_1.data(),
                      weight_1_size / 2 * sizeof(float),
                      XPUMemcpyKind::XPU_HOST_TO_DEVICE));
  XPU_CALL(xpu_memcpy(transpose_weight_1_addr + weight_1_size / 4 * 3,
                      cpu_transpose_weight_1.data() + weight_1_size / 4 * 3,
                      weight_1_size / 4 * sizeof(float),
                      XPUMemcpyKind::XPU_HOST_TO_DEVICE));

  // change bias_0 from [b_gx, b_ix, b_fx, b_ox] to [b_ix, b_fx, b_gx, b_ox]
  const float* bias_0 = param.bias_0->template data<float>();
  int bias_0_size = param.bias_0->numel();
  bias_0_ = TargetWrapperXPU::MallocScratchPad(bias_0_size * sizeof(float));
  float* bias_0_addr = reinterpret_cast<float*>(bias_0_->addr_);
  XPU_CALL(xpu_memcpy(bias_0_addr,
                      bias_0 + bias_0_size / 4,
                      bias_0_size / 4 * sizeof(float),
                      XPUMemcpyKind::XPU_HOST_TO_DEVICE));
  XPU_CALL(xpu_memcpy(bias_0_addr + bias_0_size / 4,
                      bias_0 + bias_0_size / 2,
                      bias_0_size / 4 * sizeof(float),
                      XPUMemcpyKind::XPU_HOST_TO_DEVICE));
  XPU_CALL(xpu_memcpy(bias_0_addr + bias_0_size / 2,
                      bias_0,
                      bias_0_size / 2 * sizeof(float),
                      XPUMemcpyKind::XPU_HOST_TO_DEVICE));
  XPU_CALL(xpu_memcpy(bias_0_addr + bias_0_size / 4 * 3,
                      bias_0 + bias_0_size / 4 * 3,
                      bias_0_size / 4 * sizeof(float),
                      XPUMemcpyKind::XPU_HOST_TO_DEVICE));

  // change bias_1 from [b_gx, b_ix, b_fx, b_ox] to [b_ix, b_fx, b_gx, b_ox]
  const float* bias_1 = param.bias_1->template data<float>();
  int bias_1_size = param.bias_1->numel();
  bias_1_ = TargetWrapperXPU::MallocScratchPad(bias_1_size * sizeof(float));
  float* bias_1_addr = reinterpret_cast<float*>(bias_1_->addr_);
  XPU_CALL(xpu_memcpy(bias_1_addr,
                      bias_1 + bias_1_size / 4,
                      bias_1_size / 4 * sizeof(float),
                      XPUMemcpyKind::XPU_HOST_TO_DEVICE));
  XPU_CALL(xpu_memcpy(bias_1_addr + bias_1_size / 4,
                      bias_1 + bias_1_size / 2,
                      bias_1_size / 4 * sizeof(float),
                      XPUMemcpyKind::XPU_HOST_TO_DEVICE));
  XPU_CALL(xpu_memcpy(bias_1_addr + bias_1_size / 2,
                      bias_1,
                      bias_1_size / 2 * sizeof(float),
                      XPUMemcpyKind::XPU_HOST_TO_DEVICE));
  XPU_CALL(xpu_memcpy(bias_1_addr + bias_1_size / 4 * 3,
                      bias_1 + bias_1_size / 4 * 3,
                      bias_1_size / 4 * sizeof(float),
                      XPUMemcpyKind::XPU_HOST_TO_DEVICE));

  // prepare max
  auto weight_0_ptr = param.weight_0->template data<float>();
  auto weight_0_len = param.weight_0->numel();
  float max_weight_0 =
      paddle::lite::xpu::math::FindMaxAbs(weight_0_ptr, weight_0_len);
  std::vector<float> max_weight_0_v(max_ptr_size, max_weight_0);
  weight_0_max_ =
      TargetWrapperXPU::MallocScratchPad(max_ptr_size * sizeof(float));
  float* weight_0_max_addr = reinterpret_cast<float*>(weight_0_max_->addr_);
  XPU_CALL(xpu_memcpy(weight_0_max_addr,
                      max_weight_0_v.data(),
                      max_ptr_size * sizeof(float),
                      XPUMemcpyKind::XPU_HOST_TO_DEVICE));

  float max_weight_1 = paddle::lite::xpu::math::FindMaxAbs(
      param.weight_1->template data<float>(), param.weight_1->numel());
  std::vector<float> max_weight_1_v(max_ptr_size, max_weight_1);
  weight_1_max_ =
      TargetWrapperXPU::MallocScratchPad(max_ptr_size * sizeof(float));
  float* weight_1_max_addr = reinterpret_cast<float*>(weight_1_max_->addr_);
  XPU_CALL(xpu_memcpy(weight_1_max_addr,
                      max_weight_1_v.data(),
                      max_ptr_size * sizeof(float),
                      XPUMemcpyKind::XPU_HOST_TO_DEVICE));
}

void XPUDynamicLstmCompute::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();
  bool is_reverse = param.is_reverse;
  const float* input_addr = param.input->template data<float>();

  // prepare input batch tensor
  paddle::lite::LoD in_lods = param.input->lod();
  std::vector<int> int_lod(in_lods[0].begin(), in_lods[0].end());
  std::vector<int> seq_len_tensor(in_lods[0].size() - 1);
  std::vector<int64_t> seq_len_tensor_64(in_lods[0].size() - 1);
  int max_seq_len = 0;
  for (int i = 0; i < int_lod.size() - 1; i++) {
    seq_len_tensor[i] = int_lod[i + 1] - int_lod[i];
    seq_len_tensor_64[i] = int_lod[i + 1] - int_lod[i];
    max_seq_len = std::max(max_seq_len, seq_len_tensor[i]);
  }
  int batch_size = seq_len_tensor.size();
  auto& input_dims = param.input->dims();
  int xdim = static_cast<int>(input_dims[1]);
  auto in_batch_tensor = TargetWrapperXPU::MallocScratchPad(
      max_seq_len * batch_size * xdim * sizeof(float));
  float* in_batch_tensor_addr =
      reinterpret_cast<float*>(in_batch_tensor->addr_);

  // reverse input if is_reverse = true
  if (is_reverse) {
    auto reverse_input = TargetWrapperXPU::MallocScratchPad(
        param.input->numel() * sizeof(float));
    float* reverse_input_addr = reinterpret_cast<float*>(reverse_input->addr_);
    int r = xdnn::sequence_reverse<float, int>(
        ctx.GetRawContext(),
        input_addr,
        reverse_input_addr,
        {int_lod.data(), static_cast<int>(int_lod.size()), nullptr},
        xdim);
    CHECK_EQ(r, 0);
    r = xdnn::sequence_pad<float, int>(
        ctx.GetRawContext(),
        reverse_input_addr,
        in_batch_tensor_addr,
        {int_lod.data(), static_cast<int>(int_lod.size()), nullptr},
        batch_size,
        max_seq_len,
        xdim,
        0);
    CHECK_EQ(r, 0);
  } else {
    int r = xdnn::sequence_pad<float, int>(
        ctx.GetRawContext(),
        input_addr,
        in_batch_tensor_addr,
        {int_lod.data(), static_cast<int>(int_lod.size()), nullptr},
        batch_size,
        max_seq_len,
        xdim,
        0);
    CHECK_EQ(r, 0);
  }

  // transpose from in_batch_tensor[batch_size, seq_len, xdim] to
  // transpose_in[seq_len, batch_size, xdim]
  auto transpose_in = TargetWrapperXPU::MallocScratchPad(
      batch_size * max_seq_len * xdim * sizeof(float));
  float* transpose_in_addr = reinterpret_cast<float*>(transpose_in->addr_);
  int r = xdnn::transpose<float>(ctx.GetRawContext(),
                                 in_batch_tensor_addr,
                                 transpose_in_addr,
                                 {batch_size, max_seq_len, xdim},
                                 {1, 0, 2});
  CHECK_EQ(r, 0);

  const float* h0 = param.has_h0 ? param.h0->template data<float>() : nullptr;
  const float* c0 = param.has_h0 ? param.c0->template data<float>() : nullptr;
  const float* transpose_weight_0_addr =
      reinterpret_cast<float*>(transpose_weight_0_->addr_);
  const float* transpose_weight_1_addr =
      reinterpret_cast<float*>(transpose_weight_1_->addr_);
  const float* bias_0_addr = reinterpret_cast<float*>(bias_0_->addr_);
  const float* bias_1_addr = reinterpret_cast<float*>(bias_1_->addr_);

  // prepare transpose_out tensor
  DDimLite output_dims(
      std::vector<int64_t>{param.input->dims()[0], param.weight_1->dims()[0]});
  param.hidden->Resize(output_dims);
  int hdim = static_cast<int>(output_dims[1]);
  auto transpose_out = TargetWrapperXPU::MallocScratchPad(
      max_seq_len * batch_size * hdim * sizeof(float));
  float* transpose_out_addr = reinterpret_cast<float*>(transpose_out->addr_);
  auto last_h =
      TargetWrapperXPU::MallocScratchPad(batch_size * hdim * sizeof(float));
  float* last_h_addr = reinterpret_cast<float*>(last_h->addr_);
  auto last_c =
      TargetWrapperXPU::MallocScratchPad(batch_size * hdim * sizeof(float));
  float* last_c_addr = reinterpret_cast<float*>(last_c->addr_);

  const float* weight_0_maxptr = reinterpret_cast<float*>(weight_0_max_->addr_);
  const float* weight_1_maxptr = reinterpret_cast<float*>(weight_1_max_->addr_);

  auto x_seq_len_guard =
      TargetWrapperXPU::MallocScratchPad(batch_size * sizeof(int64_t));
  int64_t* x_seq_len = reinterpret_cast<int64_t*>(x_seq_len_guard->addr_);
  XPU_CALL(xpu_memcpy(x_seq_len,
                      seq_len_tensor_64.data(),
                      batch_size * sizeof(int64_t),
                      XPUMemcpyKind::XPU_HOST_TO_DEVICE));
  r = xdnn::lstm_inference(ctx.GetRawContext(),
                           max_seq_len,
                           batch_size,
                           xdim,
                           hdim,
                           false,
                           transpose_in_addr,
                           h0,
                           c0,
                           x_seq_len,
                           transpose_weight_0_addr,
                           weight_0_maxptr,
                           transpose_weight_1_addr,
                           weight_1_maxptr,
                           bias_0_addr,
                           bias_1_addr,
                           transpose_out_addr,
                           last_h_addr,
                           last_c_addr);
  CHECK_EQ(r, 0);

  // transpose from transpose_out[seq_len, batch_size, hdim] to
  // out_batch_tensor[batch_size, seq_len, hdim]
  auto out_batch_tensor = TargetWrapperXPU::MallocScratchPad(
      max_seq_len * batch_size * hdim * sizeof(float));
  float* out_batch_tensor_addr =
      reinterpret_cast<float*>(out_batch_tensor->addr_);
  r = xdnn::transpose<float>(ctx.GetRawContext(),
                             transpose_out_addr,
                             out_batch_tensor_addr,
                             {max_seq_len, batch_size, hdim},
                             {1, 0, 2});
  CHECK_EQ(r, 0);

  if (is_reverse) {
    auto reverse_output = TargetWrapperXPU::MallocScratchPad(
        param.hidden->numel() * sizeof(float));
    float* reverse_output_addr =
        reinterpret_cast<float*>(reverse_output->addr_);

    r = xdnn::sequence_unpad<float, int>(
        ctx.GetRawContext(),
        out_batch_tensor_addr,
        reverse_output_addr,
        {int_lod.data(), static_cast<int>(int_lod.size()), nullptr},
        max_seq_len,
        hdim);
    CHECK_EQ(r, 0);

    r = xdnn::sequence_reverse<float, int>(
        ctx.GetRawContext(),
        reverse_output_addr,
        param.hidden->template mutable_data<float>(TARGET(kXPU)),
        {int_lod.data(), static_cast<int>(int_lod.size()), nullptr},
        hdim);
    CHECK_EQ(r, 0);
  } else {
    r = xdnn::sequence_unpad<float, int>(
        ctx.GetRawContext(),
        out_batch_tensor_addr,
        param.hidden->template mutable_data<float>(TARGET(kXPU)),
        {int_lod.data(), static_cast<int>(int_lod.size()), nullptr},
        max_seq_len,
        hdim);
    CHECK_EQ(r, 0);
  }
  param.hidden->set_lod(param.input->lod());
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

namespace xpu = paddle::lite::kernels::xpu;
using XPUDynamicLstmFp = xpu::XPUDynamicLstmCompute;

REGISTER_LITE_KERNEL(
    __xpu__dynamic_lstm_fuse_op, kXPU, kFloat, kNCHW, XPUDynamicLstmFp, def)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Weight_0", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindInput("Weight_1", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindInput("Bias_0", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindInput("Bias_1", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindInput("H0", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("C0", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Hidden", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();
