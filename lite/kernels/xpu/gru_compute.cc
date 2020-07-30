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
#include <string>
#include <vector>
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

struct SeqInfo {
  SeqInfo() = default;
  SeqInfo(int start, int length, int seq_idx)
      : start(start), length(length), seq_idx(seq_idx) {}
  int start;
  int length;
  int seq_idx;
};

inline xdnn::Activation_t get_gru_act_type(const std::string& type) {
  if (type == "sigmoid") {
    return xdnn::Activation_t::SIGMOID; /* type */
  } else if (type == "tanh") {
    return xdnn::Activation_t::TANH; /* type */
  } else if (type == "relu") {
    return xdnn::Activation_t::RELU; /* type */
  } else {
    LOG(FATAL) << "unsupported activation type: " << type;
  }
}

void GruCompute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->As<XPUContext>();
  // inputs
  bool origin_mode = param.origin_mode;
  auto input = param.input;
  int frame_size = input->dims()[1] / 3;
  //-------- sequence 2 batch -------//
  auto lods = input->lod();
  const auto& lod = lods[0];
  // offset_cpu
  int* offset_cpu = reinterpret_cast<int*>(malloc((lod.size()) * sizeof(int)));
  for (auto i = 0; i < lod.size(); i++) {
    offset_cpu[i] = lod[i];
  }
  // idx_sorted_by_width_data_cpu
  std::vector<SeqInfo> seq_info(lod.size() - 1);
  for (size_t seq_id = 0; seq_id < lod.size() - 1; ++seq_id) {
    int length = lod[seq_id + 1] - lod[seq_id];
    seq_info[seq_id].start = lod[seq_id];
    seq_info[seq_id].length = length;
    seq_info[seq_id].seq_idx = seq_id;
  }
  std::stable_sort(seq_info.begin(), seq_info.end(), [](SeqInfo a, SeqInfo b) {
    return a.length > b.length;
  });
  int* idx_sorted_by_width_data_cpu =
      reinterpret_cast<int*>(malloc(seq_info.size() * sizeof(int)));
  for (auto i = 0; i < seq_info.size(); i++) {
    idx_sorted_by_width_data_cpu[i] = seq_info[i].seq_idx;
  }
  // max_width
  int max_width = seq_info[0].length;
  // new_offset_cpu
  int* new_offset_cpu =
      reinterpret_cast<int*>(malloc((max_width + 1) * sizeof(int)));
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
  // offset_xpu
  int* offset_xpu;
  xpu_malloc(reinterpret_cast<void**>(&offset_xpu), lod.size() * sizeof(int));
  xpu_memcpy(
      offset_xpu, offset_cpu, sizeof(int) * lod.size(), XPU_HOST_TO_DEVICE);
  // idx_sorted_by_width_data_cpu
  int* idx_sorted_by_width_data_xpu;
  xpu_malloc(reinterpret_cast<void**>(&idx_sorted_by_width_data_xpu),
             seq_info.size() * sizeof(int));
  xpu_memcpy(idx_sorted_by_width_data_xpu,
             idx_sorted_by_width_data_cpu,
             sizeof(int) * seq_info.size(),
             XPU_HOST_TO_DEVICE);
  // new_offset_xpu
  int* new_offset_xpu;
  xpu_malloc(reinterpret_cast<void**>(&new_offset_xpu),
             (max_width + 1) * sizeof(int));
  xpu_memcpy(new_offset_xpu,
             new_offset_cpu,
             sizeof(int) * (max_width + 1),
             XPU_HOST_TO_DEVICE);
  // batch_data
  float* batch_data;
  xpu_malloc(reinterpret_cast<void**>(&batch_data),
             lod[lod.size() - 1] * frame_size * 3 * sizeof(float));
  // xpu kernel
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
  //-------- sequence 2 batch -------//

  // perpare xpu_h_p
  auto* h0 = param.h0;
  float* xpu_h0_start;
  xpu_malloc(reinterpret_cast<void**>(&xpu_h0_start),
             (lod.size() - 1) * frame_size * sizeof(float));
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

  float* batch_gate = param.batch_gate->mutable_data<float>(TARGET(kXPU));
  float* batch_reset_hidden_prev =
      param.batch_reset_hidden_prev->mutable_data<float>(TARGET(kXPU));
  float* batch_hidden = param.hidden->mutable_data<float>(TARGET(kXPU));

  float* weight_ur_max_ptr_xpu;
  float weight_ur_max_cpu[4];
  xpu_malloc(reinterpret_cast<void**>(&weight_ur_max_ptr_xpu),
             4 * sizeof(float));
  int ret = xdnn::findmax(ctx.GetRawContext(),
                          param.weight->data<float>(),
                          frame_size * frame_size * 2,
                          weight_ur_max_ptr_xpu);
  CHECK_EQ(ret, 0);
  xpu_memcpy(weight_ur_max_cpu,
             weight_ur_max_ptr_xpu,
             sizeof(float) * 4,
             XPU_DEVICE_TO_HOST);
  float weight_u_r_max_value =
      std::max(std::max(weight_ur_max_cpu[0], weight_ur_max_cpu[1]),
               std::max(weight_ur_max_cpu[2], weight_ur_max_cpu[3]));

  float* weight_c_max_ptr_xpu;
  float weight_c_max_cpu[4];
  xpu_malloc(reinterpret_cast<void**>(&weight_c_max_ptr_xpu),
             4 * sizeof(float));
  ret = xdnn::findmax(ctx.GetRawContext(),
                      param.weight->data<float>() + frame_size * frame_size * 2,
                      frame_size * frame_size,
                      weight_c_max_ptr_xpu);
  CHECK_EQ(ret, 0);
  xpu_memcpy(weight_c_max_cpu,
             weight_c_max_ptr_xpu,
             sizeof(float) * 4,
             XPU_DEVICE_TO_HOST);
  float weight_c_max_value =
      std::max(std::max(weight_c_max_cpu[0], weight_c_max_cpu[1]),
               std::max(weight_c_max_cpu[2], weight_c_max_cpu[3]));

  for (int batch_idx = 0; batch_idx < max_width; batch_idx++) {
    float* x = batch_data + new_offset_cpu[batch_idx] * frame_size * 3;

    int ret = xdnn::gru_unit_int31(
        ctx.GetRawContext(),
        new_offset_cpu[batch_idx + 1] - new_offset_cpu[batch_idx],
        frame_size,
        origin_mode,
        get_gru_act_type(param.gate_activation),
        get_gru_act_type(param.activation),
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
        xdnn::sequence_reverse(ctx.GetRawContext(), /* context */
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

  xpu_free(weight_ur_max_ptr_xpu);
  xpu_free(weight_c_max_ptr_xpu);
  xpu_free(offset_xpu);
  xpu_free(idx_sorted_by_width_data_xpu);
  xpu_free(new_offset_xpu);
  free(offset_cpu);
  free(idx_sorted_by_width_data_cpu);
  free(new_offset_cpu);
  xpu_free(batch_data);
  xpu_free(xpu_h0_start);
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
