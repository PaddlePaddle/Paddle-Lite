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

#include <memory>
#include <vector>
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/kernel.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

void fill_max(float max, float* xpu_ptr) {
  float maxs[4] = {max, 0.0f, 0.0f, 0.0f};
  xpu_memcpy(
      xpu_ptr, maxs, 4 * sizeof(float), XPUMemcpyKind::XPU_HOST_TO_DEVICE);
}

void grnn_layout(int batch,
                 const std::vector<int>& offset,
                 std::vector<int>* new_offset_ptr,
                 std::vector<int>* idx_sorted_ptr) {
  auto& new_offset = *new_offset_ptr;
  auto& idx_sorted = *idx_sorted_ptr;

  std::vector<int> width;
  width.resize(batch);
  new_offset.clear();
  idx_sorted.clear();

  idx_sorted.resize(batch);
  for (int i = 0; i < batch; i++) {
    width[i] = offset[i + 1] - offset[i];
    idx_sorted[i] = i;
  }
  std::sort(idx_sorted.data(),
            idx_sorted.data() + batch,
            [&width](int a, int b) { return width[a] > width[b]; });
  int max_width = width[idx_sorted[0]];
  new_offset.resize(max_width + 1);
  new_offset[0] = 0;
  int j = batch - 1;
  int last_width = 0;
  int sub_row = 0;
  int sub_col = 0;

  for (int i = 1; i <= max_width;) {
    for (int k = j; k >= 0; --k) {
      if (width[idx_sorted[k]] > last_width) {
        sub_row = width[idx_sorted[k]] - last_width;
        sub_col = k + 1;
        for (int s = 0; s < sub_row; s++) {
          new_offset[i] = new_offset[i - 1] + sub_col;
          i++;
        }
        // move on
        last_width = width[idx_sorted[k]];
        j = k - 1;
        break;
      }
    }
  }
}

class id_info {
  XPUScratchPadGuard l3_buffer_guard;
  char* l3_buffer;
  std::unique_ptr<char[]> cpu_buffer_guard;
  char* cpu_buffer;

 public:
  const int64_t* id0_64;
  const int64_t* id1_64;
  int64_t* lod_64;
  int* lod_32;
  int* new_offset_32;
  int* idx_sorted_32;

  std::vector<int> lod;
  std::vector<int> new_offset;
  std::vector<int> idx_sorted;
  int batch;
  int seqlen_max;
  int seqlen_sum;
  int seqlen_square_sum;

  void init(int UB_batch, int UB_seqlen) {
    int ub_lod_64_size = (UB_batch + 1) * sizeof(int64_t);
    int ub_lod_32_size = (UB_batch + 1) * sizeof(int);
    int ub_new_offset_32_size = (UB_seqlen + 1) * sizeof(int);
    int ub_idx_sorted_32_size = (UB_batch + 1) * sizeof(int);
    int total_size = ub_lod_64_size + ub_lod_32_size + ub_new_offset_32_size +
                     ub_idx_sorted_32_size;

    // TODO(miaotianxiang): use l3?
    l3_buffer_guard = TargetWrapperXPU::MallocScratchPad(total_size, false);
    l3_buffer = reinterpret_cast<char*>(l3_buffer_guard->addr_);
    cpu_buffer_guard.reset(new char[total_size]);
    cpu_buffer = cpu_buffer_guard.get();
  }

  void update(lite::Tensor* _id0, lite::Tensor* _id1) {
    auto& id0_lod = _id0->lod()[0];
    lod.clear();
    for (auto e : id0_lod) {
      lod.push_back(e);
    }

    seqlen_max = 0;
    seqlen_sum = 0;
    seqlen_square_sum = 0;
    batch = lod.size() - 1;
    for (int i = 0; i < batch; i++) {
      int seqlen = lod[i + 1] - lod[i];
      seqlen_max = std::max(seqlen_max, seqlen);
      seqlen_sum = seqlen_sum + seqlen;
      seqlen_square_sum = seqlen_square_sum + seqlen * seqlen;
    }
    grnn_layout(batch, lod, &new_offset, &idx_sorted);

    id0_64 = _id0->data<int64_t>();
    id1_64 = _id1->data<int64_t>();

    int offset = 0;
    lod_64 = reinterpret_cast<int64_t*>(l3_buffer + offset);
    memcpy(
        cpu_buffer + offset, id0_lod.data(), id0_lod.size() * sizeof(int64_t));
    offset += id0_lod.size() * sizeof(int64_t);
    lod_32 = reinterpret_cast<int*>(l3_buffer + offset);
    memcpy(cpu_buffer + offset, lod.data(), lod.size() * sizeof(int));
    offset += lod.size() * sizeof(int);
    new_offset_32 = reinterpret_cast<int*>(l3_buffer + offset);
    memcpy(cpu_buffer + offset,
           new_offset.data(),
           new_offset.size() * sizeof(int));
    offset += new_offset.size() * sizeof(int);
    idx_sorted_32 = reinterpret_cast<int*>(l3_buffer + offset);
    memcpy(cpu_buffer + offset,
           idx_sorted.data(),
           idx_sorted.size() * sizeof(int));
    offset += idx_sorted.size() * sizeof(int);
    xpu_memcpy(
        l3_buffer, cpu_buffer, offset, XPUMemcpyKind::XPU_HOST_TO_DEVICE);
  }
};

class fc_op {
  const int16_t* weight;
  XPUScratchPadGuard weight_max_guard;
  float* weight_max;
  const float* bias;
  XPUScratchPadGuard in_max_guard;
  float* in_max;
  int n;
  int k;
  xdnn::Activation_t::act_enum act_type;
  XPUScratchPadGuard out_max_guard;

 public:
  float* out_max;

  void init(const int16_t* _weight,
            float _weight_max,
            const float* _bias,
            int _n,
            int _k,
            xdnn::Activation_t::act_enum _act_type) {
    n = _n;
    k = _k;
    act_type = _act_type;

    weight = _weight;
    weight_max_guard =
        TargetWrapperXPU::MallocScratchPad(4 * sizeof(float), false);
    weight_max = reinterpret_cast<float*>(weight_max_guard->addr_);
    fill_max(_weight_max, weight_max);

    bias = _bias;

    in_max_guard = TargetWrapperXPU::MallocScratchPad(4 * sizeof(float), false);
    out_max_guard =
        TargetWrapperXPU::MallocScratchPad(4 * sizeof(float), false);
    in_max = reinterpret_cast<float*>(in_max_guard->addr_);
    out_max = reinterpret_cast<float*>(in_max_guard->addr_);
  }

  void init(lite::Tensor* _weight,
            float _weight_max,
            lite::Tensor* _bias,
            int _n,
            int _k,
            xdnn::Activation_t::act_enum _act_type) {
    init(_weight->data<int16_t>(),
         _weight_max,
         _bias ? _bias->data<float>() : nullptr,
         _n,
         _k,
         _act_type);
  }

  void infer(xdnn::Context* ctx,
             const float* in,
             int m,
             float* out,
             const float* in_max_by_caller = nullptr) {
    if (in_max_by_caller == nullptr) {
      xdnn::findmax<float>(ctx, in, m * k, in_max);
      in_max_by_caller = in_max;
    }
    xdnn::gemm_int16_maxptr<float, int16_t, float>(ctx,
                                                   false,
                                                   true,
                                                   m,
                                                   n,
                                                   k,
                                                   1.0f,
                                                   in,
                                                   k,
                                                   weight,
                                                   k,
                                                   0.0f,
                                                   out,
                                                   n,
                                                   bias,
                                                   act_type,
                                                   in_max_by_caller,
                                                   weight_max,
                                                   out_max);
  }
};

class grnn_op {
  fc_op fc_e2h0;
  fc_op fc_e2h1;
  fc_op fc_e2h2;
  const int16_t* dense_h2h;
  float dense_h2h_max[3];
  XPUScratchPadGuard input_max_guard;
  float* input_max;
  XPUScratchPadGuard hbm_buffer_guard;
  float* hbm_buffer;
  // require: cap_l * max(cap_e, cap_h) * 5
  // seq2batch_out: [cap_l, cap_e]
  // fc_e2h_out: [3, cap_l, cap_h]
  // gru_out: [cap_l, cap_h]
  int cap_e;
  int cap_h;
  int max_cap_l;

 public:
  void init(lite::Tensor* _wh,
            const std::vector<float>& _wh_maxs,
            lite::Tensor* _wi,
            const std::vector<float>& _wi_maxs,
            int _cap_e,
            int _cap_h,
            int _max_cap_l) {
    cap_e = _cap_e;
    cap_h = _cap_h;
    max_cap_l = _max_cap_l;

    // weight
    auto* dense_e2h = _wi->data<int16_t>();
    fc_e2h0.init(dense_e2h,
                 _wi_maxs[0],
                 nullptr,
                 cap_h,
                 cap_e,
                 xdnn::Activation_t::LINEAR);
    fc_e2h1.init(dense_e2h + cap_e * cap_h,
                 _wi_maxs[1],
                 nullptr,
                 cap_h,
                 cap_e,
                 xdnn::Activation_t::LINEAR);
    fc_e2h2.init(dense_e2h + cap_e * cap_h * 2,
                 _wi_maxs[2],
                 nullptr,
                 cap_h,
                 cap_e,
                 xdnn::Activation_t::LINEAR);

    dense_h2h = _wh->data<int16_t>();
    dense_h2h_max[0] = _wh_maxs[0];
    dense_h2h_max[1] = _wh_maxs[1];
    dense_h2h_max[2] = _wh_maxs[2];

    input_max_guard =
        TargetWrapperXPU::MallocScratchPad(4 * sizeof(float), false);
    input_max = reinterpret_cast<float*>(input_max_guard->addr_);
    hbm_buffer_guard = TargetWrapperXPU::MallocScratchPad(
        5 * std::max(cap_e, cap_h) * max_cap_l * sizeof(float), false);
    hbm_buffer = reinterpret_cast<float*>(hbm_buffer_guard->addr_);
  }

  void infer(xdnn::Context* ctx,
             const id_info& sentense,
             const float* in,
             float* out,
             float* l3_buffer = nullptr,
             int l3_size = 0) {
    int batch = sentense.batch;
    int cap_l = sentense.seqlen_sum;
    int max_width = sentense.seqlen_max;

    int slot_size = cap_l * std::max(cap_e, cap_h);
    float* seq2batch_out = hbm_buffer;
    float* fc_e2h_out = hbm_buffer + 1 * slot_size;
    float* gru_out = hbm_buffer + 4 * slot_size;
    if (l3_size > 0 && l3_size >= 5 * slot_size * sizeof(float)) {
      seq2batch_out = l3_buffer;
      fc_e2h_out = l3_buffer + 1 * slot_size;
      gru_out = l3_buffer + 4 * slot_size;
    }

    xdnn::search_seq2batch(ctx,
                           batch,
                           max_width,
                           cap_e,
                           sentense.idx_sorted_32,
                           sentense.lod_32,
                           sentense.new_offset_32,
                           in,
                           seq2batch_out);

    xdnn::findmax<float>(ctx, in, cap_l * cap_e, input_max);
    fc_e2h0.infer(ctx, seq2batch_out, cap_l, fc_e2h_out, input_max);
    fc_e2h1.infer(
        ctx, seq2batch_out, cap_l, fc_e2h_out + cap_l * cap_h, input_max);
    fc_e2h2.infer(
        ctx, seq2batch_out, cap_l, fc_e2h_out + cap_l * cap_h * 2, input_max);
    xdnn::search_grnn<float, int16_t>(ctx,
                                      cap_l,
                                      cap_h,
                                      cap_e,
                                      max_width,
                                      sentense.new_offset_32,
                                      fc_e2h_out,
                                      dense_h2h,
                                      gru_out,
                                      dense_h2h_max[0],
                                      dense_h2h_max[1],
                                      dense_h2h_max[2]);

    xdnn::search_batch2seq(ctx,
                           batch,
                           max_width,
                           cap_h,
                           sentense.idx_sorted_32,
                           sentense.lod_32,
                           sentense.new_offset_32,
                           gru_out,
                           out);
  }
};

class attention_op {
  int dim;
  float alpha0;
  float alpha1;
  fc_op seqfc;
  XPUScratchPadGuard hbm_buffer_guard;
  float* hbm_buffer;
  // require: cap_l * dim + seqlen_square_sum
  // seqfc_out: [cap_l, dim]
  // batchgemm0_out: [seqlen_square_sum]
  // seq_softmax_out: [seqlen_square_sum], reuse of batchgemm0_out
  // batchgemm1_out: [cap_l, dim], reuse of seqfc_out

 public:
  void init(lite::Tensor* _att_fc_w,
            float _att_fc_w_max,
            lite::Tensor* _att_fc_b,
            int _dim,
            int UB_batch,
            int UB_seqlen) {
    dim = _dim;
    alpha0 = 0.0883883461356163f;  // TODO(miaotianxiang):
    alpha1 = 1.0f;

    seqfc.init(_att_fc_w,
               _att_fc_w_max,
               _att_fc_b,
               dim,
               dim,
               xdnn::Activation_t::LINEAR);
    hbm_buffer_guard = TargetWrapperXPU::MallocScratchPad(
        (UB_batch * (UB_seqlen * dim + UB_seqlen * UB_seqlen)) * sizeof(float),
        false);
    hbm_buffer = reinterpret_cast<float*>(hbm_buffer_guard->addr_);
  }

  void infer(xdnn::Context* ctx,
             const id_info& sentense,
             const float* input,
             float* pool_out,
             float* l3_buffer = nullptr,
             int l3_size = 0) {
    int batch = sentense.batch;
    int cap_l = sentense.seqlen_sum;
    int max_width = sentense.seqlen_max;
    int* lod_32 = sentense.lod_32;

    float* seqfc_out = hbm_buffer;
    float* batchgemm0_out = hbm_buffer + cap_l * dim;
    float* seq_softmax_out = batchgemm0_out;
    float* batchgemm1_out = seqfc_out;
    if (l3_size > 0 &&
        l3_size >= (cap_l * dim + sentense.seqlen_square_sum) * sizeof(float)) {
      seqfc_out = l3_buffer;
      batchgemm0_out = l3_buffer + cap_l * dim;
      seq_softmax_out = batchgemm0_out;
      batchgemm1_out = seqfc_out;
    }

    seqfc.infer(ctx, input, cap_l, seqfc_out);
    xdnn::search_noaligned_mat_mul(ctx,
                                   0,
                                   1,
                                   batch,
                                   lod_32,
                                   max_width,
                                   dim,
                                   alpha0,
                                   input,
                                   seqfc_out,
                                   batchgemm0_out);
    xdnn::search_seq_softmax(
        ctx, batchgemm0_out, seq_softmax_out, lod_32, batch, max_width);
    xdnn::search_noaligned_mat_mul(ctx,
                                   0,
                                   0,
                                   batch,
                                   lod_32,
                                   max_width,
                                   dim,
                                   alpha1,
                                   seq_softmax_out,
                                   input,
                                   batchgemm1_out);
    xdnn::sequence_pooling_forward(ctx,
                                   xdnn::Pooling_t::MAX_WITHOUT_INDEX,
                                   batch,
                                   lod_32,
                                   dim,
                                   batchgemm1_out,
                                   nullptr,
                                   pool_out);
  }
};

class match_conv_topk {
  std::vector<int> topks;
  int dim_t;
  int dim_in;
  int out_channel;

  fc_op xw_fc;
  const int16_t* conv_weight;
  float conv_weight_max;
  XPUScratchPadGuard hbm_buffer_guard;
  float* hbm_buffer;
  // xw_out: [sum(left_len), dim_t * dim_in]
  // xwy_out: [sum(left_len *right_len) * dim_t]
  // conv_out: [sum(left_len *right_len) * out_channel]
  // seq_concat_out: [sum(left_len *right_len) * (dim_t + out_channel)]

  XPUScratchPadGuard left_lod_32_guard;
  int* left_lod_32;
  XPUScratchPadGuard right_lod_32_guard;
  int* right_lod_32;
  XPUScratchPadGuard match_lod_32_guard;
  int* match_lod_32;
  XPUScratchPadGuard conv_lod_32_guard;
  int* conv_lod_32;
  XPUScratchPadGuard topk_offset_32_guard;
  int* topk_offset_32;
  XPUScratchPadGuard topks_xpu_guard;
  int* topks_xpu;
  XPUScratchPadGuard useless_topk_pos_guard;
  int* useless_topk_pos;

 public:
  float* seq_avg_topk_out;

  void init(lite::Tensor* _input_w,
            float _input_w_max,
            lite::Tensor* _conv_w,
            float _conv_w_max,
            int _dim_t,
            int _dim_in,
            int UB_batch,
            int UB_seqlen,
            const std::vector<int>& _topks) {
    dim_t = _dim_t;
    dim_in = _dim_in;
    out_channel = 5;  // TODO(miaotianxiang):
    topks = _topks;

    xw_fc.init(_input_w,
               _input_w_max,
               nullptr,
               dim_t * dim_in,
               dim_in,
               xdnn::Activation_t::LINEAR);
    conv_weight = _conv_w->data<int16_t>();
    conv_weight_max = _conv_w_max;

    hbm_buffer_guard = TargetWrapperXPU::MallocScratchPad(
        (UB_batch * UB_seqlen * dim_t * dim_in +
         UB_batch * UB_seqlen * UB_seqlen * (dim_t + out_channel) * 2) *
            sizeof(float),
        false);
    hbm_buffer = reinterpret_cast<float*>(hbm_buffer_guard->addr_);

    left_lod_32_guard =
        TargetWrapperXPU::MallocScratchPad((UB_batch + 1) * sizeof(int), false);
    left_lod_32 = reinterpret_cast<int*>(left_lod_32_guard->addr_);
    right_lod_32_guard =
        TargetWrapperXPU::MallocScratchPad((UB_batch + 1) * sizeof(int), false);
    right_lod_32 = reinterpret_cast<int*>(right_lod_32_guard->addr_);
    match_lod_32_guard =
        TargetWrapperXPU::MallocScratchPad((UB_batch + 1) * sizeof(int), false);
    match_lod_32 = reinterpret_cast<int*>(match_lod_32_guard->addr_);
    conv_lod_32_guard =
        TargetWrapperXPU::MallocScratchPad((UB_batch + 1) * sizeof(int), false);
    conv_lod_32 = reinterpret_cast<int*>(conv_lod_32_guard->addr_);
    topk_offset_32_guard =
        TargetWrapperXPU::MallocScratchPad((UB_batch + 1) * sizeof(int), false);
    topk_offset_32 = reinterpret_cast<int*>(topk_offset_32_guard->addr_);
    topks_xpu_guard =
        TargetWrapperXPU::MallocScratchPad(topks.size() * sizeof(int), false);
    topks_xpu = reinterpret_cast<int*>(topks_xpu_guard->addr_);
    xpu_memcpy(topks_xpu,
               topks.data(),
               topks.size() * sizeof(int),
               XPUMemcpyKind::XPU_HOST_TO_DEVICE);
    useless_topk_pos_guard =
        TargetWrapperXPU::MallocScratchPad(4 * sizeof(int), false);
    useless_topk_pos = reinterpret_cast<int*>(useless_topk_pos_guard->addr_);
  }

  void infer(xdnn::Context* ctx,
             lite::Tensor* left,
             lite::Tensor* right,
             lite::Tensor* out,
             float* l3_buffer = nullptr,
             int l3_size = 0) {
    auto left_lod = left->lod()[0];
    auto right_lod = right->lod()[0];
    int batch = left_lod.size() - 1;

    std::vector<int> left_lod_32_cpu;
    for (auto e : left_lod) {
      left_lod_32_cpu.push_back(e);
    }
    xpu_memcpy(left_lod_32,
               left_lod_32_cpu.data(),
               left_lod_32_cpu.size() * sizeof(int),
               XPUMemcpyKind::XPU_HOST_TO_DEVICE);
    std::vector<int> right_lod_32_cpu;
    for (auto e : right_lod) {
      right_lod_32_cpu.push_back(e);
    }
    xpu_memcpy(right_lod_32,
               right_lod_32_cpu.data(),
               right_lod_32_cpu.size() * sizeof(int),
               XPUMemcpyKind::XPU_HOST_TO_DEVICE);

    std::vector<int> lod_match = {0};
    std::vector<int> lod_conv = {0};
    std::vector<int> lod_topk = {0};
    int x_mul_y_sum = 0;
    int left_seqlen_sum = 0;
    int left_seqlen_max = 0;
    int right_seqlen_sum = 0;
    int right_seqlen_max = 0;
    for (int i = 0; i < batch; i++) {
      int len_x = left_lod[i + 1] - left_lod[i];
      int len_y = right_lod[i + 1] - right_lod[i];
      int imgsize = len_x * len_y;
      x_mul_y_sum = x_mul_y_sum + imgsize;
      lod_match.push_back(lod_match.back() + imgsize * dim_t);
      lod_conv.push_back(lod_conv.back() + imgsize * out_channel);
      lod_topk.push_back(lod_topk.back() + imgsize * (dim_t + out_channel));

      left_seqlen_max = std::max(left_seqlen_max, len_x);
      right_seqlen_max = std::max(right_seqlen_max, len_y);
      left_seqlen_sum += len_x;
      right_seqlen_sum += len_y;
    }
    xpu_memcpy(match_lod_32,
               lod_match.data(),
               lod_match.size() * sizeof(int),
               XPUMemcpyKind::XPU_HOST_TO_DEVICE);
    xpu_memcpy(conv_lod_32,
               lod_conv.data(),
               lod_conv.size() * sizeof(int),
               XPUMemcpyKind::XPU_HOST_TO_DEVICE);
    xpu_memcpy(topk_offset_32,
               lod_topk.data(),
               lod_topk.size() * sizeof(int),
               XPUMemcpyKind::XPU_HOST_TO_DEVICE);

    float* xwy_out = hbm_buffer;
    float* conv_out = hbm_buffer + x_mul_y_sum * dim_t;
    float* seq_concat_out = hbm_buffer + x_mul_y_sum * (dim_t + out_channel);
    float* xw_out = hbm_buffer + x_mul_y_sum * (dim_t + out_channel) * 2;
    int total_len = x_mul_y_sum * (dim_t + out_channel) * 2 +
                    left_seqlen_sum * dim_t * dim_in;
    if (l3_size > 0 && l3_size >= total_len * sizeof(float)) {
      xwy_out = l3_buffer;
      conv_out = l3_buffer + x_mul_y_sum * dim_t;
      seq_concat_out = l3_buffer + x_mul_y_sum * (dim_t + out_channel);
      xw_out = l3_buffer + x_mul_y_sum * (dim_t + out_channel) * 2;
    }
    seq_avg_topk_out = out->mutable_data<float>(TARGET(kXPU));

    int max_width = std::max(left_seqlen_max, right_seqlen_max);
    xw_fc.infer(ctx, left->data<float>(), left_seqlen_sum, xw_out);
    xdnn::match_matrix_tensor(ctx,
                              batch,
                              xw_out,
                              right->data<float>(),
                              left_lod_32,
                              right_lod_32,
                              dim_t,
                              dim_in,
                              xwy_out,
                              xw_fc.out_max,
                              xdnn::Activation_t::RELU,
                              max_width);
    xdnn::search_varconv<float, int16_t>(
        ctx,
        batch,
        dim_t,
        out_channel,
        5,
        5,
        1,
        1,
        xwy_out,
        conv_weight,
        right_lod_32,
        left_lod_32,
        conv_out,
        conv_weight_max,
        xdnn::Activation_t::RELU);  // TODO(miaotianxiang):
    xdnn::sequence_concat(ctx,
                          xwy_out,
                          match_lod_32,
                          conv_out,
                          conv_lod_32,
                          seq_concat_out,
                          batch);
    xdnn::sequence_topk_avg_pooling(ctx,
                                    seq_concat_out,
                                    seq_avg_topk_out,
                                    useless_topk_pos,
                                    batch,
                                    dim_t + out_channel,
                                    topk_offset_32,
                                    left_lod_32,
                                    right_lod_32,
                                    topks_xpu,
                                    topks.size());
  }
};

class bid_emb_grnn_att {
  const float* table;
  int table_len;
  int emb_dim;
  int cap_h;
  grnn_op bi_fw;
  grnn_op bi_rv;
  attention_op att;
  XPUScratchPadGuard hbm_buffer_guard;
  float* hbm_buffer;
  // require at least: 4 * cap_l * emb_dim
  // emb_rv: [cap_l, emb_dim]
  // grnn_fw: [cap_l, emb_dim]
  // grnn_rv: [cap_l, emb_dim]
  // grnn_rv_rv: [cap_l, emb_dim]
  // concat_2in: [cap_l, 2 * emb_dim]
  // L3.bi_fw: 5 * cap_l * emb_dim
  // L3.bi_rv: 5 * cap_l * emb_dim
  // L3.att:   cap_l * 2 * emb_dim + seqlen_square_sum

  // execution-plan:
  // 1. bid_emb_ew,                   alloc(emb_rv)
  // 2. bi_rv,                        alloc(grnn_rv)
  // 3.                               free(emb_rv)
  // 4. sequence_reverse,             alloc(grnn_rv_rv)
  // 5. sequence_pooling(grnn_rv)
  // 6.                               free(grnn_rv)
  // 7. bi_fw                         alloc(grnn_fw)
  // 8. sequence_pooling(grnn_fw)
  // 9. concat_2                      alloc(concat_2in)
  // 10. concat_3
  // 11. att

  // alloc-plan:
  // [0]: emb_rv, grnn_rv_rv
  // [1]: grnn_rv, grnn_fw
  // [2, 3]: concat_2in
  // [2, 3, 4, 5, 6]: L3.bi_fw, L3.bi_rv
  // [4, 5, ..., ?]:  L3.att

 public:
  float* emb_fw;
  float* concat_3in;
  float* pool_fw;
  float* pool_rv;
  float* att_out;

  void init(lite::Tensor* _table,
            lite::Tensor* _fw_wh,
            const std::vector<float>& _fw_wh_maxs,
            lite::Tensor* _fw_wi,
            const std::vector<float>& _fw_wi_maxs,
            lite::Tensor* _rv_wh,
            const std::vector<float>& _rv_wh_maxs,
            lite::Tensor* _rv_wi,
            const std::vector<float>& _rv_wi_maxs,
            lite::Tensor* _att_fc_w,
            float _att_fc_w_max,
            lite::Tensor* _att_fc_b,
            int UB_batch,
            int UB_seqlen) {
    table = _table->data<float>();
    table_len = _table->dims()[0];
    emb_dim = _table->dims()[1];
    cap_h = emb_dim;
    int max_cap_l = UB_batch * UB_seqlen;

    bi_fw.init(
        _fw_wh, _fw_wh_maxs, _fw_wi, _fw_wi_maxs, emb_dim, cap_h, max_cap_l);
    bi_rv.init(
        _rv_wh, _rv_wh_maxs, _rv_wi, _rv_wi_maxs, emb_dim, cap_h, max_cap_l);
    att.init(
        _att_fc_w, _att_fc_w_max, _att_fc_b, 2 * cap_h, UB_batch, UB_seqlen);

    hbm_buffer_guard = TargetWrapperXPU::MallocScratchPad(
        4 * max_cap_l * cap_h * sizeof(float), false);
    hbm_buffer = reinterpret_cast<float*>(hbm_buffer_guard->addr_);
  }

  void infer(xdnn::Context* ctx,
             int batch,
             const id_info& sentense,
             lite::Tensor* grnn_fw_pool_out,
             lite::Tensor* grnn_rv_pool_out,
             lite::Tensor* att_pool_out,
             lite::Tensor* concat_3in1_out,
             lite::Tensor* emb_fw_out,
             float* l3_buffer = nullptr,
             int l3_size = 0) {
    int cap_l = sentense.seqlen_sum;
    int slot_len = cap_l * cap_h;

    float* emb_rv = hbm_buffer;
    float* grnn_fw = hbm_buffer + slot_len;
    float* grnn_rv = hbm_buffer + slot_len;
    float* grnn_rv_rv = hbm_buffer;
    float* concat_2in = hbm_buffer + 2 * slot_len;
    if (l3_size > 0 && l3_size >= 4 * slot_len * sizeof(float)) {
      emb_rv = l3_buffer;
      grnn_fw = l3_buffer + slot_len;
      grnn_rv = l3_buffer + slot_len;
      grnn_rv_rv = l3_buffer;
    }
    emb_fw = emb_fw_out->mutable_data<float>(TARGET(kXPU));
    concat_3in = concat_3in1_out->mutable_data<float>(TARGET(kXPU));
    pool_fw = grnn_fw_pool_out->mutable_data<float>(TARGET(kXPU));
    pool_rv = grnn_rv_pool_out->mutable_data<float>(TARGET(kXPU));
    att_out = att_pool_out->mutable_data<float>(TARGET(kXPU));

    xdnn::search_bid_emb_ew(ctx,
                            batch,
                            sentense.lod_64,
                            sentense.id0_64,
                            sentense.id1_64,
                            table,
                            table_len,
                            emb_dim,
                            emb_fw,
                            emb_rv,
                            table_len - 2,
                            1);
    bi_rv.infer(ctx,
                sentense,
                emb_rv,
                grnn_rv,
                l3_buffer + 2 * slot_len,
                l3_size - 2 * slot_len * sizeof(float));
    xdnn::sequence_reverse(
        ctx, batch, sentense.lod_32, cap_h, grnn_rv, grnn_rv_rv);
    xdnn::sequence_pooling_forward(ctx,
                                   xdnn::Pooling_t::LAST,
                                   batch,
                                   sentense.lod_32,
                                   cap_h,
                                   grnn_rv,
                                   nullptr,
                                   pool_rv);

    bi_fw.infer(ctx,
                sentense,
                emb_fw,
                grnn_fw,
                l3_buffer + 2 * slot_len,
                l3_size - 2 * slot_len * sizeof(float));
    xdnn::sequence_pooling_forward(ctx,
                                   xdnn::Pooling_t::LAST,
                                   batch,
                                   sentense.lod_32,
                                   cap_h,
                                   grnn_fw,
                                   nullptr,
                                   pool_fw);
    const int concat_widths[] = {cap_h, cap_h, cap_h};
    const float* concat_ptrs[] = {emb_fw, grnn_fw, grnn_rv_rv};
    xdnn::concat<float>(
        ctx, cap_l, concat_widths + 1, 2, concat_ptrs + 1, concat_2in);
    xdnn::concat<float>(ctx, cap_l, concat_widths, 3, concat_ptrs, concat_3in);
    att.infer(ctx,
              sentense,
              concat_2in,
              att_out,
              l3_buffer + 4 * slot_len,
              l3_size - 4 * slot_len * sizeof(float));
  }
};

class emb_att {
  const float* table;
  int table_len;
  int emb_dim;
  attention_op att;

 public:
  float* emb_fw;
  float* att_out;

  void init(lite::Tensor* _table,
            lite::Tensor* _att_fc_w,
            float _att_fc_w_max,
            lite::Tensor* _att_fc_b,
            int UB_batch,
            int UB_seqlen) {
    table = _table->data<float>();
    table_len = _table->dims()[0];
    emb_dim = _table->dims()[1];
    att.init(_att_fc_w, _att_fc_w_max, _att_fc_b, emb_dim, UB_batch, UB_seqlen);
  }

  void infer(xdnn::Context* ctx,
             int batch,
             const id_info& sentense,
             lite::Tensor* att_pool_out,
             lite::Tensor* emb_fw_out,
             float* l3_buffer = nullptr,
             int l3_size = 0) {
    emb_fw = emb_fw_out->mutable_data<float>(TARGET(kXPU));
    att_out = att_pool_out->mutable_data<float>(TARGET(kXPU));

    int cap_l = sentense.lod.back();
    const float* emb_tables[] = {table, table};
    const int64_t* emb_indices[] = {sentense.id0_64, sentense.id1_64};
    xdnn::embedding_with_ewadd<float, int64_t, false, false>(ctx,
                                                             emb_dim,
                                                             cap_l,
                                                             2,
                                                             table_len - 2,
                                                             emb_tables,
                                                             emb_indices,
                                                             nullptr,
                                                             nullptr,
                                                             emb_fw);
    att.infer(ctx, sentense, emb_fw, att_out, l3_buffer, l3_size);
  }
};

class merge_all {
  grnn_op coverage_fw;
  grnn_op coverage_rv;
  int cap_e;
  int cap_h;

  // TODO(miaotianxiang):
  const int fc0_k = 1152;
  const int fc0_n = 512;
  const int fc1_k = 640;
  const int fc1_n = 320;
  const int fc2_k = 320;
  const int fc2_n = 1;
  fc_op fc0;
  fc_op fc1;
  fc_op fc2;

  XPUScratchPadGuard hbm_buffer_guard;
  float* hbm_buffer;
  // topk_concat_out_fw:  [cap_l, cap_e] <= [cap_l, cap_h]
  // topk_concat_out_rv:  [cap_l, cap_e] <= [cap_l, cap_h]
  // grnn_fw:             [cap_l, cap_h]
  // grnn_rv:             [cap_l, cap_h]
  // pool_fw:             [batch, cap_h]
  // pool_rv:             [batch, cap_h]
  // fc0_in:              [batch, fc0_k]
  // fc0_out:             [batch, fc0_n]
  // fc1_in:              [batch, fc1_k]
  // fc1_out:             [batch, fc1_n]
  // fc2_out:             [batch, fc2_n]

 public:
  void init(lite::Tensor* grnn_fw_wh,
            std::vector<float> grnn_fw_wh_maxs,
            lite::Tensor* grnn_fw_wi,
            std::vector<float> grnn_fw_wi_maxs,
            lite::Tensor* grnn_rv_wh,
            std::vector<float> grnn_rv_wh_maxs,
            lite::Tensor* grnn_rv_wi,
            std::vector<float> grnn_rv_wi_maxs,
            lite::Tensor* fc0_w,
            float fc0_w_max,
            lite::Tensor* fc0_b,
            lite::Tensor* fc1_w,
            float fc1_w_max,
            lite::Tensor* fc1_b,
            lite::Tensor* fc2_w,
            float fc2_w_max,
            lite::Tensor* fc2_b,
            int UB_batch,
            int UB_seqlen) {
    int max_cap_l = UB_batch * UB_seqlen;
    cap_e = grnn_fw_wi->dims()[2];
    cap_h = grnn_fw_wi->dims()[1];

    coverage_fw.init(grnn_fw_wh,
                     grnn_fw_wh_maxs,
                     grnn_fw_wi,
                     grnn_fw_wi_maxs,
                     cap_e,
                     cap_h,
                     max_cap_l);
    coverage_rv.init(grnn_rv_wh,
                     grnn_rv_wh_maxs,
                     grnn_rv_wi,
                     grnn_rv_wi_maxs,
                     cap_e,
                     cap_h,
                     max_cap_l);

    fc0.init(fc0_w, fc0_w_max, fc0_b, fc0_n, fc0_k, xdnn::Activation_t::RELU);
    fc1.init(fc1_w, fc1_w_max, fc1_b, fc1_n, fc1_k, xdnn::Activation_t::RELU);
    fc2.init(fc2_w, fc2_w_max, fc2_b, fc2_n, fc2_k, xdnn::Activation_t::LINEAR);

    int hbm_total_len =
        max_cap_l * cap_h * 4 +
        UB_batch * (2 * cap_h + fc0_k + fc0_n + fc1_k + fc1_n + fc2_n);
    hbm_buffer_guard = TargetWrapperXPU::MallocScratchPad(
        hbm_total_len * sizeof(float), false);
    hbm_buffer = reinterpret_cast<float*>(hbm_buffer_guard->addr_);
  }

  void infer(xdnn::Context* ctx,
             const id_info& sentense,
             const std::vector<lite::Tensor*> concat_2in1_x,
             const std::vector<lite::Tensor*> concat_7in1_x,
             lite::Tensor* out,
             float* l3_buffer = nullptr,
             int l3_size = 0) {
    int batch = sentense.batch;
    int cap_l = sentense.seqlen_sum;

    float* topk_concat_out_fw = hbm_buffer;
    int hbm_total_len =
        cap_l * cap_h * 4 +
        batch * (2 * cap_h + fc0_k + fc0_n + fc1_k + fc1_n + fc2_n);
    if (l3_size > 0 && l3_size >= hbm_total_len * sizeof(float)) {
      topk_concat_out_fw = l3_buffer;
    }
    float* topk_concat_out_rv = topk_concat_out_fw + cap_l * cap_h;
    float* grnn_fw = topk_concat_out_rv + cap_l * cap_h;
    float* grnn_rv = grnn_fw + cap_l * cap_h;
    float* pool_fw = grnn_rv + cap_l * cap_h;
    float* pool_rv = pool_fw + batch * cap_h;
    float* fc0_in = pool_fw + batch * cap_h * 2;
    float* fc0_out = fc0_in + batch * fc0_k;
    float* fc1_in = fc0_out + batch * fc0_n;
    float* fc1_out = fc1_in + batch * fc1_k;
    // float* fc2_out = fc1_out + batch * fc1_n;
    float* fc2_out = out->mutable_data<float>(TARGET(kXPU));

    const int concat_widths[] = {static_cast<int>(concat_2in1_x[0]->dims()[1]),
                                 static_cast<int>(concat_2in1_x[1]->dims()[1])};
    const float* concat_ptrs[] = {concat_2in1_x[0]->data<float>(),
                                  concat_2in1_x[1]->data<float>()};
    xdnn::concat<float>(
        ctx, cap_l, concat_widths, 2, concat_ptrs, topk_concat_out_fw);
    xdnn::sequence_reverse(ctx,
                           batch,
                           sentense.lod_32,
                           cap_e,
                           topk_concat_out_fw,
                           topk_concat_out_rv);
    coverage_fw.infer(ctx,
                      sentense,
                      topk_concat_out_fw,
                      grnn_fw,
                      l3_buffer + hbm_total_len,
                      l3_size - hbm_total_len * sizeof(float));
    coverage_rv.infer(ctx,
                      sentense,
                      topk_concat_out_rv,
                      grnn_rv,
                      l3_buffer + hbm_total_len,
                      l3_size - hbm_total_len * sizeof(float));
    xdnn::sequence_pooling_forward(ctx,
                                   xdnn::Pooling_t::LAST,
                                   batch,
                                   sentense.lod_32,
                                   cap_h,
                                   grnn_fw,
                                   nullptr,
                                   pool_fw);
    xdnn::sequence_pooling_forward(ctx,
                                   xdnn::Pooling_t::LAST,
                                   batch,
                                   sentense.lod_32,
                                   cap_h,
                                   grnn_rv,
                                   nullptr,
                                   pool_rv);

    const int concat_widths_fc0[] = {
        static_cast<int>(concat_7in1_x[0]->dims()[1]),
        static_cast<int>(concat_7in1_x[1]->dims()[1]),
        static_cast<int>(concat_7in1_x[2]->dims()[1]),
        static_cast<int>(concat_7in1_x[3]->dims()[1]),
        static_cast<int>(concat_7in1_x[4]->dims()[1]),
        static_cast<int>(concat_7in1_x[5]->dims()[1]),
        static_cast<int>(concat_7in1_x[6]->dims()[1]),
    };
    const float* concat_ptrs_fc0[] = {
        concat_7in1_x[0]->data<float>(),
        concat_7in1_x[1]->data<float>(),
        concat_7in1_x[2]->data<float>(),
        concat_7in1_x[3]->data<float>(),
        concat_7in1_x[4]->data<float>(),
        concat_7in1_x[5]->data<float>(),
        concat_7in1_x[6]->data<float>(),
    };
    const int concat_widths_fc1[] = {cap_h, cap_h, fc0_n};
    const float* concat_ptrs_fc1[] = {pool_fw, pool_rv, fc0_out};

    xdnn::concat<float>(
        ctx, batch, concat_widths_fc0, 7, concat_ptrs_fc0, fc0_in);
    fc0.infer(ctx, fc0_in, batch, fc0_out);
    xdnn::concat<float>(
        ctx, batch, concat_widths_fc1, 3, concat_ptrs_fc1, fc1_in);
    fc1.infer(ctx, fc1_in, batch, fc1_out);
    fc2.infer(ctx, fc1_out, batch, fc2_out);
  }
};

class XPUMMDNNBidEmbGrnnAttCompute
    : public KernelLite<TARGET(kXPU), PRECISION(kFloat)> {
 public:
  using param_t = operators::XPUMMDNNBidEmbGrnnAttParam;

  void PrepareForRun() override;

  void Run() override;

 private:
  id_info id;
  bid_emb_grnn_att compound;
  int UB_batch = 40;
  int UB_seqlen = 512;
};

void XPUMMDNNBidEmbGrnnAttCompute::PrepareForRun() {
  auto& param = this->Param<param_t>();

  id.init(UB_batch, UB_seqlen);
  compound.init(param.emb_tbl,
                param.grnn_fw_wh,
                param.grnn_fw_wh_maxs,
                param.grnn_fw_wi,
                param.grnn_fw_wi_maxs,
                param.grnn_rv_wh,
                param.grnn_rv_wh_maxs,
                param.grnn_rv_wi,
                param.grnn_rv_wi_maxs,
                param.att_fc_w,
                param.att_fc_w_max,
                param.att_fc_b,
                UB_batch,
                UB_seqlen);
}

void XPUMMDNNBidEmbGrnnAttCompute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->As<XPUContext>();

  auto* xpu_ctx = ctx.GetRawContext();

  int batch = param.id0->lod()[0].size() - 1;
  id.update(param.id0, param.id1);
  compound.infer(ctx.GetRawContext(),
                 batch,
                 id,
                 param.grnn_fw_pool_out,
                 param.grnn_rv_pool_out,
                 param.att_pool_out,
                 param.concat_3in1_out,
                 param.emb_fw_out,
                 reinterpret_cast<float*>(
                     reinterpret_cast<char*>(xpu_ctx->workspace_l3_ptr) +
                     xpu_ctx->used_l3_size),
                 xpu_ctx->workspace_l3_size - xpu_ctx->used_l3_size);
}

class XPUMMDNNBidEmbAttCompute
    : public KernelLite<TARGET(kXPU), PRECISION(kFloat)> {
 public:
  using param_t = operators::XPUMMDNNBidEmbAttParam;

  void PrepareForRun() override;

  void Run() override;

 private:
  id_info id;
  emb_att compound;
  int UB_batch = 40;
  int UB_seqlen = 512;
};

void XPUMMDNNBidEmbAttCompute::PrepareForRun() {
  auto& param = this->Param<param_t>();

  id.init(UB_batch, UB_seqlen);
  compound.init(param.emb_tbl,
                param.att_fc_w,
                param.att_fc_w_max,
                param.att_fc_b,
                UB_batch,
                UB_seqlen);
}

void XPUMMDNNBidEmbAttCompute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->As<XPUContext>();

  auto* xpu_ctx = ctx.GetRawContext();

  int batch = param.id0->lod()[0].size() - 1;
  id.update(param.id0, param.id1);
  compound.infer(ctx.GetRawContext(),
                 batch,
                 id,
                 param.att_pool_out,
                 param.emb_fw_out,
                 reinterpret_cast<float*>(
                     reinterpret_cast<char*>(xpu_ctx->workspace_l3_ptr) +
                     xpu_ctx->used_l3_size),
                 xpu_ctx->workspace_l3_size - xpu_ctx->used_l3_size);
}

class XPUMMDNNMatchConvTopkCompute
    : public KernelLite<TARGET(kXPU), PRECISION(kFloat)> {
 public:
  using param_t = operators::XPUMMDNNMatchConvTopkParam;

  void PrepareForRun() override;

  void Run() override;

 private:
  match_conv_topk compound;
  int UB_batch = 40;
  int UB_seqlen = 512;
};

void XPUMMDNNMatchConvTopkCompute::PrepareForRun() {
  auto& param = this->Param<param_t>();

  compound.init(param.input_w,
                param.input_w_max,
                param.conv_w,
                param.conv_w_max,
                param.dim_t,
                param.input_w->dims()[0],
                UB_batch,
                UB_seqlen,
                param.topks);
}

void XPUMMDNNMatchConvTopkCompute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->As<XPUContext>();

  auto* xpu_ctx = ctx.GetRawContext();

  compound.infer(ctx.GetRawContext(),
                 param.input_x,
                 param.input_y,
                 param.topk_out,
                 reinterpret_cast<float*>(
                     reinterpret_cast<char*>(xpu_ctx->workspace_l3_ptr) +
                     xpu_ctx->used_l3_size),
                 xpu_ctx->workspace_l3_size - xpu_ctx->used_l3_size);
}

class XPUMMDNNMergeAllCompute
    : public KernelLite<TARGET(kXPU), PRECISION(kFloat)> {
 public:
  using param_t = operators::XPUMMDNNMergeAllParam;

  void PrepareForRun() override;

  void Run() override;

 private:
  id_info id;
  merge_all compound;
  int UB_batch = 40;
  int UB_seqlen = 512;
};

void XPUMMDNNMergeAllCompute::PrepareForRun() {
  auto& param = this->Param<param_t>();

  id.init(UB_batch, UB_seqlen);
  compound.init(param.grnn_fw_wh,
                param.grnn_fw_wh_maxs,
                param.grnn_fw_wi,
                param.grnn_fw_wi_maxs,
                param.grnn_rv_wh,
                param.grnn_rv_wh_maxs,
                param.grnn_rv_wi,
                param.grnn_rv_wi_maxs,
                param.fc0_w,
                param.fc0_w_max,
                param.fc0_b,
                param.fc1_w,
                param.fc1_w_max,
                param.fc1_b,
                param.fc2_w,
                param.fc2_w_max,
                param.fc2_b,
                UB_batch,
                UB_seqlen);
}

void XPUMMDNNMergeAllCompute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->As<XPUContext>();

  auto* xpu_ctx = ctx.GetRawContext();

  id.update(param.concat_2in1_x[0], param.concat_2in1_x[1]);
  compound.infer(ctx.GetRawContext(),
                 id,
                 param.concat_2in1_x,
                 param.concat_7in1_x,
                 param.out,
                 reinterpret_cast<float*>(
                     reinterpret_cast<char*>(xpu_ctx->workspace_l3_ptr) +
                     xpu_ctx->used_l3_size),
                 xpu_ctx->workspace_l3_size - xpu_ctx->used_l3_size);
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(__xpu__mmdnn_bid_emb_grnn_att,
                     kXPU,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::xpu::XPUMMDNNBidEmbGrnnAttCompute,
                     def)
    .BindInput("id0", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt64))})
    .BindInput("id1", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt64))})
    .BindInput("emb_tbl", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("grnn_fw_wh", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("grnn_fw_wi", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("grnn_rv_wh", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("grnn_rv_wi", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("att_fc_w", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("att_fc_b", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("grnn_fw_pool_out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("grnn_rv_pool_out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("att_pool_out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("concat_3in1_out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("emb_fw_out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();

REGISTER_LITE_KERNEL(__xpu__mmdnn_bid_emb_att,
                     kXPU,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::xpu::XPUMMDNNBidEmbAttCompute,
                     def)
    .BindInput("id0", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt64))})
    .BindInput("id1", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt64))})
    .BindInput("emb_tbl", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("att_fc_w", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("att_fc_b", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("att_pool_out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("concat_3in1_out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("emb_fw_out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();

REGISTER_LITE_KERNEL(__xpu__mmdnn_match_conv_topk,
                     kXPU,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::xpu::XPUMMDNNMatchConvTopkCompute,
                     def)
    .BindInput("input_x", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("input_y", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("input_w", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("conv_w", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("topk_out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();

REGISTER_LITE_KERNEL(__xpu__mmdnn_merge_all,
                     kXPU,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::xpu::XPUMMDNNMergeAllCompute,
                     def)
    .BindInput("concat_7in1_x", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("concat_2in1_x", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("grnn_fw_wh", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("grnn_fw_wi", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("grnn_rv_wh", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("grnn_rv_wi", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("fc0_w", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("fc0_b", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("fc1_w", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("fc1_b", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("fc2_w", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("fc2_b", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();
