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

#include "lite/kernels/xpu/gru_compute.h"
#include <math.h>
#include <algorithm>
#include <map>
#include <string>
#include <vector>
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

inline xdnn::Activation_t GetGruActType(const std::string& type) {
  std::map<std::string, xdnn::Activation_t> act_type_map = {
      {"sigmoid", xdnn::Activation_t::SIGMOID},
      {"tanh", xdnn::Activation_t::TANH},
      {"relu", xdnn::Activation_t::RELU}};
  auto it = act_type_map.find(type);
  if (it != act_type_map.end()) {
    return it->second;
  } else {
    LOG(FATAL) << "unsupported activation type: " << type;
    return xdnn::Activation_t(xdnn::Activation_t::act_enum(0));
  }
}

void GruCompute::PrepareForRun() {
  offset_xpu_guard_ = TargetWrapperXPU::MallocScratchPad(
      XPU_MAX_LOD_SIZE * sizeof(int), false /* use_l3 */);
  new_offset_xpu_guard_ = TargetWrapperXPU::MallocScratchPad(
      XPU_MAX_LOD_SEQ_LEN * sizeof(int), false /* use_l3 */);
  idx_sorted_by_width_data_xpu_guard_ = TargetWrapperXPU::MallocScratchPad(
      XPU_MAX_LOD_SIZE * sizeof(int), false /* use_l3 */);

  idx_sorted_by_width_data_cpu.reset(new int[XPU_MAX_LOD_SIZE]);
  offset_cpu.reset(new int[XPU_MAX_LOD_SIZE]);
  new_offset_cpu.reset(new int[XPU_MAX_LOD_SEQ_LEN]);

  // find max
  maxs_xpu_guard_ =
      TargetWrapperXPU::MallocScratchPad(8 * sizeof(float), false /* use_l3 */);
  auto& ctx = this->ctx_->As<XPUContext>();
  auto& param = this->Param<param_t>();
  int frame_size = param.input->dims()[1] / 3;
  float* weight_ur_max_ptr_xpu =
      reinterpret_cast<float*>(maxs_xpu_guard_->addr_);
  float* weight_c_max_ptr_xpu = weight_ur_max_ptr_xpu + 4;

  // weight_ur_max
  int ret = xdnn::findmax(ctx.GetRawContext(),
                          param.weight->data<float>(),
                          frame_size * frame_size * 2,
                          weight_ur_max_ptr_xpu);
  CHECK_EQ(ret, 0);
  // weight_c_max
  ret = xdnn::findmax(ctx.GetRawContext(),
                      param.weight->data<float>() + frame_size * frame_size * 2,
                      frame_size * frame_size,
                      weight_c_max_ptr_xpu);
  CHECK_EQ(ret, 0);

  float weight_ur_max_cpu[4];
  XPU_CALL(xpu_memcpy(weight_ur_max_cpu,
                      weight_ur_max_ptr_xpu,
                      sizeof(float) * 4,
                      XPUMemcpyKind::XPU_DEVICE_TO_HOST));
  weight_u_r_max_value =
      std::max(std::max(weight_ur_max_cpu[0], weight_ur_max_cpu[1]),
               std::max(weight_ur_max_cpu[2], weight_ur_max_cpu[3]));

  float weight_c_max_cpu[4];
  XPU_CALL(xpu_memcpy(weight_c_max_cpu,
                      weight_c_max_ptr_xpu,
                      sizeof(float) * 4,
                      XPUMemcpyKind::XPU_DEVICE_TO_HOST));
  weight_c_max_value =
      std::max(std::max(weight_c_max_cpu[0], weight_c_max_cpu[1]),
               std::max(weight_c_max_cpu[2], weight_c_max_cpu[3]));
}

void GruCompute::PrepareLayout(const paddle::lite::LoD& lods,
                               int* offset_xpu,
                               int* new_offset_xpu,
                               int* idx_sorted_by_width_data_xpu) {
  const auto& lod = lods[0];
  for (auto i = 0; i < lod.size(); i++) {
    offset_cpu[i] = lod[i];
  }
  for (size_t seq_id = 0; seq_id < lod.size() - 1; ++seq_id) {
    int length = lod[seq_id + 1] - lod[seq_id];
    seq_info.push_back(SeqInfo(lod[seq_id], length, seq_id));
  }
  std::cout << "seq len is " << seq_info.size() << std::endl;
  std::stable_sort(seq_info.begin(), seq_info.end(), [](SeqInfo a, SeqInfo b) {
    return a.length > b.length;
  });
  for (auto i = 0; i < seq_info.size(); i++) {
    idx_sorted_by_width_data_cpu[i] = seq_info[i].seq_idx;
  }
  // max_width
  int max_width = seq_info[0].length;
  new_offset_cpu[0] = 0;
  int cur_offset_idx = 1;
  for (auto i = 0; i < seq_info.size(); i++) {
    int cur_length = seq_info.size() - i;
    int repeat_times = (i == 0) ? seq_info[i].length
                                : (seq_info[i].length - seq_info[i - 1].length);
    for (int j = 0; j < repeat_times; j++) {
      new_offset_cpu[cur_offset_idx] =
          new_offset_cpu[cur_offset_idx - 1] + cur_length;
      cur_offset_idx++;
    }
  }
  XPU_CALL(xpu_memcpy(offset_xpu,
                      offset_cpu.get(),
                      sizeof(int) * lod.size(),
                      XPU_HOST_TO_DEVICE));

  XPU_CALL(xpu_memcpy(idx_sorted_by_width_data_xpu,
                      idx_sorted_by_width_data_cpu.get(),
                      sizeof(int) * seq_info.size(),
                      XPU_HOST_TO_DEVICE));

  XPU_CALL(xpu_memcpy(new_offset_xpu,
                      new_offset_cpu.get(),
                      sizeof(int) * (max_width + 1),
                      XPU_HOST_TO_DEVICE));
}

void GruCompute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->As<XPUContext>();

  auto input = param.input;
  float* batch_gate = param.batch_gate->mutable_data<float>(TARGET(kXPU));
  float* batch_reset_hidden_prev =
      param.batch_reset_hidden_prev->mutable_data<float>(TARGET(kXPU));
  float* batch_hidden = param.hidden->mutable_data<float>(TARGET(kXPU));
  bool origin_mode = param.origin_mode;
  int frame_size = input->dims()[1] / 3;

  int* offset_xpu = reinterpret_cast<int*>(offset_xpu_guard_->addr_);
  int* new_offset_xpu = reinterpret_cast<int*>(new_offset_xpu_guard_->addr_);
  int* idx_sorted_by_width_data_xpu =
      reinterpret_cast<int*>(idx_sorted_by_width_data_xpu_guard_->addr_);

  // prepare seq_info
  auto lods = input->lod();
  const auto& lod = lods[0];
  PrepareLayout(lods, offset_xpu, new_offset_xpu, idx_sorted_by_width_data_xpu);
  int max_width = seq_info[0].length;

  // sequence to batch
  XPUScratchPadGuard xpu_batch_data_guard_ = TargetWrapperXPU::MallocScratchPad(
      lod[lod.size() - 1] * frame_size * 3 * sizeof(float), false /*use_l3 */);
  float* batch_data = reinterpret_cast<float*>(xpu_batch_data_guard_->addr_);

  bool is_reverse = param.is_reverse;
  if (is_reverse) {
    int ret = xdnn::sequence_reverse(ctx.GetRawContext(), /* context */
                                     lod.size() - 1,
                                     offset_xpu,
                                     frame_size * 3,
                                     param.input->data<float>(),
                                     batch_data);
    CHECK_EQ(ret, 0);
    ret = xdnn::search_seq2batch(ctx.GetRawContext(), /* context */
                                 lod.size() - 1,
                                 max_width,
                                 frame_size * 3,
                                 idx_sorted_by_width_data_xpu,
                                 offset_xpu,
                                 new_offset_xpu,
                                 batch_data,
                                 batch_data);
    CHECK_EQ(ret, 0);
  } else {
    int ret = xdnn::search_seq2batch(ctx.GetRawContext(), /* context */
                                     lod.size() - 1,
                                     max_width,
                                     frame_size * 3,
                                     idx_sorted_by_width_data_xpu,
                                     offset_xpu,
                                     new_offset_xpu,
                                     param.input->data<float>(),
                                     batch_data);
    CHECK_EQ(ret, 0);
  }
  // perpare xpu_h_p
  auto* h0 = param.h0;
  XPUScratchPadGuard xpu_h0_guard_ = TargetWrapperXPU::MallocScratchPad(
      (lod.size() - 1) * frame_size * sizeof(float), false /*use_l3 */);
  float* xpu_h0_start = reinterpret_cast<float*>(xpu_h0_guard_->addr_);
  float* xpu_h0 = xpu_h0_start;
  if (h0) {
    for (auto i = 0; i < seq_info.size(); i++) {
      int ret = xdnn::memcpy_device(
          ctx.GetRawContext(),
          xpu_h0 + i * frame_size,
          h0->data<float>() + seq_info[i].seq_idx * frame_size,
          sizeof(float) * frame_size);
      CHECK_EQ(ret, 0);
    }
  } else {
    // initial with zero
    int ret = xdnn::scale(ctx.GetRawContext(),
                          frame_size * seq_info.size(),
                          0.0,
                          0.0,
                          false,
                          xpu_h0,
                          xpu_h0);
    CHECK_EQ(ret, 0);
  }
  // gru
  for (int batch_idx = 0; batch_idx < max_width; batch_idx++) {
    float* x = batch_data + new_offset_cpu[batch_idx] * frame_size * 3;

    int ret = xdnn::gru_unit_int31(
        ctx.GetRawContext(),
        new_offset_cpu[batch_idx + 1] - new_offset_cpu[batch_idx],
        frame_size,
        origin_mode,
        GetGruActType(param.gate_activation),
        GetGruActType(param.activation),
        x,
        xpu_h0,
        param.weight->data<float>(),
        weight_u_r_max_value,
        weight_c_max_value,
        param.bias->data<float>(),
        batch_gate + new_offset_cpu[batch_idx] * frame_size * 3,
        batch_reset_hidden_prev + new_offset_cpu[batch_idx] * frame_size,
        batch_hidden + new_offset_cpu[batch_idx] * frame_size);

    CHECK_EQ(ret, 0);
    xpu_h0 = batch_hidden + new_offset_cpu[batch_idx] * frame_size;
  }
  // batch to sequence
  if (is_reverse) {
    int ret = xdnn::search_batch2seq(ctx.GetRawContext(),
                                     seq_info.size(),
                                     max_width,
                                     frame_size,
                                     idx_sorted_by_width_data_xpu,
                                     offset_xpu,
                                     new_offset_xpu,
                                     batch_hidden,
                                     batch_data);
    CHECK_EQ(ret, 0);
    ret =
        xdnn::sequence_reverse(ctx.GetRawContext(),
                               lod.size() - 1,
                               offset_xpu,
                               frame_size,
                               batch_data,
                               param.hidden->mutable_data<float>(TARGET(kXPU)));
    CHECK_EQ(ret, 0);

  } else {
    int ret =
        xdnn::search_batch2seq(ctx.GetRawContext(),
                               seq_info.size(),
                               max_width,
                               frame_size,
                               idx_sorted_by_width_data_xpu,
                               offset_xpu,
                               new_offset_xpu,
                               batch_hidden,
                               param.hidden->mutable_data<float>(TARGET(kXPU)));
    CHECK_EQ(ret, 0);
  }
  seq_info.clear();
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(
    gru, kXPU, kFloat, kNCHW, paddle::lite::kernels::xpu::GruCompute, def)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("H0", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Weight", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("BatchGate", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("BatchResetHiddenPrev", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("BatchHidden", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Hidden", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();
