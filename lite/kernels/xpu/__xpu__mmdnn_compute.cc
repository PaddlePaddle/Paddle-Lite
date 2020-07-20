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

namespace {

void FillMax(float max, float* xpu_ptr) {
  float maxs[4] = {max, 0.0f, 0.0f, 0.0f};
  XPU_CALL(xpu_memcpy(
      xpu_ptr, maxs, 4 * sizeof(float), XPUMemcpyKind::XPU_HOST_TO_DEVICE));
}

void GrnnLayout(int batch,
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

}  // anonymous namespace

class MMDNNIdInfo {
  XPUScratchPadGuard l3_buffer_guard_;
  char* l3_buffer_{nullptr};
  std::unique_ptr<char[]> cpu_buffer_guard_;
  char* cpu_buffer_{nullptr};

 public:
  const int64_t* id0_64{nullptr};
  const int64_t* id1_64{nullptr};
  int64_t* lod_64{nullptr};
  int* lod_32{nullptr};
  int* new_offset_32{nullptr};
  int* idx_sorted_32{nullptr};

  std::vector<int> lod;
  std::vector<int> new_offset;
  std::vector<int> idx_sorted;
  int batch;
  int seqlen_max;
  int seqlen_sum;
  int seqlen_square_sum;

  void Init(int upper_bound_batch, int upper_bound_seqlen) {
    int ub_lod_64_size = (upper_bound_batch + 1) * sizeof(int64_t);
    int ub_lod_32_size = (upper_bound_batch + 1) * sizeof(int);
    int ub_new_offset_32_size = (upper_bound_seqlen + 1) * sizeof(int);
    int ub_idx_sorted_32_size = (upper_bound_batch + 1) * sizeof(int);
    int total_size = ub_lod_64_size + ub_lod_32_size + ub_new_offset_32_size +
                     ub_idx_sorted_32_size;

    // TODO(miaotianxiang): use l3?
    l3_buffer_guard_ = TargetWrapperXPU::MallocScratchPad(total_size, false);
    l3_buffer_ = reinterpret_cast<char*>(l3_buffer_guard_->addr_);
    cpu_buffer_guard_.reset(new char[total_size]);
    cpu_buffer_ = cpu_buffer_guard_.get();
  }

  void Update(lite::Tensor* id0, lite::Tensor* id1) {
    auto& id0_lod = id0->lod()[0];
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
    GrnnLayout(batch, lod, &new_offset, &idx_sorted);

    id0_64 = id0->data<int64_t>();
    id1_64 = id1->data<int64_t>();

    int offset = 0;
    lod_64 = reinterpret_cast<int64_t*>(l3_buffer_ + offset);
    memcpy(
        cpu_buffer_ + offset, id0_lod.data(), id0_lod.size() * sizeof(int64_t));
    offset += id0_lod.size() * sizeof(int64_t);
    lod_32 = reinterpret_cast<int*>(l3_buffer_ + offset);
    memcpy(cpu_buffer_ + offset, lod.data(), lod.size() * sizeof(int));
    offset += lod.size() * sizeof(int);
    new_offset_32 = reinterpret_cast<int*>(l3_buffer_ + offset);
    memcpy(cpu_buffer_ + offset,
           new_offset.data(),
           new_offset.size() * sizeof(int));
    offset += new_offset.size() * sizeof(int);
    idx_sorted_32 = reinterpret_cast<int*>(l3_buffer_ + offset);
    memcpy(cpu_buffer_ + offset,
           idx_sorted.data(),
           idx_sorted.size() * sizeof(int));
    offset += idx_sorted.size() * sizeof(int);
    XPU_CALL(xpu_memcpy(
        l3_buffer_, cpu_buffer_, offset, XPUMemcpyKind::XPU_HOST_TO_DEVICE));
  }
};

class MMDNNFcOp {
  const int16_t* weight_{nullptr};
  XPUScratchPadGuard weight_max_guard_;
  float* weight_max_{nullptr};
  const float* bias_{nullptr};
  XPUScratchPadGuard in_max_guard_;
  float* in_max_{nullptr};
  int n_;
  int k_;
  xdnn::Activation_t::act_enum act_type_;
  XPUScratchPadGuard out_max_guard_;

 public:
  float* out_max{nullptr};

  void Init(const int16_t* weight,
            float weight_max,
            const float* bias,
            int n,
            int k,
            xdnn::Activation_t::act_enum act_type) {
    n_ = n;
    k_ = k;
    act_type_ = act_type;

    weight_ = weight;
    weight_max_guard_ =
        TargetWrapperXPU::MallocScratchPad(4 * sizeof(float), false);
    weight_max_ = reinterpret_cast<float*>(weight_max_guard_->addr_);
    FillMax(weight_max, weight_max_);

    bias_ = bias;

    in_max_guard_ =
        TargetWrapperXPU::MallocScratchPad(4 * sizeof(float), false);
    out_max_guard_ =
        TargetWrapperXPU::MallocScratchPad(4 * sizeof(float), false);
    in_max_ = reinterpret_cast<float*>(in_max_guard_->addr_);
    out_max = reinterpret_cast<float*>(in_max_guard_->addr_);
  }

  void Init(lite::Tensor* weight,
            float weight_max,
            lite::Tensor* bias,
            int n,
            int k,
            xdnn::Activation_t::act_enum act_type) {
    Init(weight->data<int16_t>(),
         weight_max,
         bias ? bias->data<float>() : nullptr,
         n,
         k,
         act_type);
  }

  void Infer(xdnn::Context* ctx,
             const float* in,
             int m,
             float* out,
             const float* in_max_by_caller = nullptr) {
    int r = 0;
    if (in_max_by_caller == nullptr) {
      r = xdnn::findmax<float>(ctx, in, m * k_, in_max_);
      CHECK_EQ(r, 0);
      in_max_by_caller = in_max_;
    }
    r = xdnn::gemm_int16_maxptr<float, int16_t, float>(ctx,
                                                       false,
                                                       true,
                                                       m,
                                                       n_,
                                                       k_,
                                                       1.0f,
                                                       in,
                                                       k_,
                                                       weight_,
                                                       k_,
                                                       0.0f,
                                                       out,
                                                       n_,
                                                       bias_,
                                                       act_type_,
                                                       in_max_by_caller,
                                                       weight_max_,
                                                       out_max);
    CHECK_EQ(r, 0);
  }
};

class MMDNNGrnnOp {
  MMDNNFcOp fc_e2h0_;
  MMDNNFcOp fc_e2h1_;
  MMDNNFcOp fc_e2h2_;
  const int16_t* dense_h2h_{nullptr};
  float dense_h2h_max_[3];
  XPUScratchPadGuard input_max_guard_;
  float* input_max_{nullptr};
  XPUScratchPadGuard hbm_buffer_guard_;
  float* hbm_buffer_{nullptr};
  // require: cap_l * max(cap_e_, cap_h_) * 5
  // seq2batch_out: [cap_l, cap_e_]
  // fc_e2h_out: [3, cap_l, cap_h_]
  // gru_out: [cap_l, cap_h_]
  int cap_e_;
  int cap_h_;
  int max_cap_l_;

 public:
  void Init(lite::Tensor* wh,
            const std::vector<float>& wh_maxs,
            lite::Tensor* wi,
            const std::vector<float>& wi_maxs,
            int cap_e,
            int cap_h,
            int max_cap_l) {
    cap_e_ = cap_e;
    cap_h_ = cap_h;
    max_cap_l_ = max_cap_l;

    // weight
    auto* dense_e2h = wi->data<int16_t>();
    fc_e2h0_.Init(dense_e2h,
                  wi_maxs[0],
                  nullptr,
                  cap_h_,
                  cap_e_,
                  xdnn::Activation_t::LINEAR);
    fc_e2h1_.Init(dense_e2h + cap_e_ * cap_h_,
                  wi_maxs[1],
                  nullptr,
                  cap_h_,
                  cap_e_,
                  xdnn::Activation_t::LINEAR);
    fc_e2h2_.Init(dense_e2h + cap_e_ * cap_h_ * 2,
                  wi_maxs[2],
                  nullptr,
                  cap_h_,
                  cap_e_,
                  xdnn::Activation_t::LINEAR);

    dense_h2h_ = wh->data<int16_t>();
    dense_h2h_max_[0] = wh_maxs[0];
    dense_h2h_max_[1] = wh_maxs[1];
    dense_h2h_max_[2] = wh_maxs[2];

    input_max_guard_ =
        TargetWrapperXPU::MallocScratchPad(4 * sizeof(float), false);
    input_max_ = reinterpret_cast<float*>(input_max_guard_->addr_);
    hbm_buffer_guard_ = TargetWrapperXPU::MallocScratchPad(
        5 * std::max(cap_e_, cap_h_) * max_cap_l_ * sizeof(float), false);
    hbm_buffer_ = reinterpret_cast<float*>(hbm_buffer_guard_->addr_);
  }

  void Infer(xdnn::Context* ctx,
             const MMDNNIdInfo& sentense,
             const float* in,
             float* out,
             float* l3_buffer = nullptr,
             int l3_size = 0) {
    int batch = sentense.batch;
    int cap_l = sentense.seqlen_sum;
    int max_width = sentense.seqlen_max;

    int slot_size = cap_l * std::max(cap_e_, cap_h_);
    float* seq2batch_out = hbm_buffer_;
    float* fc_e2h_out = hbm_buffer_ + 1 * slot_size;
    float* gru_out = hbm_buffer_ + 4 * slot_size;
    if (l3_size > 0 && l3_size >= 5 * slot_size * sizeof(float)) {
      seq2batch_out = l3_buffer;
      fc_e2h_out = l3_buffer + 1 * slot_size;
      gru_out = l3_buffer + 4 * slot_size;
    }

    int r = 0;
    r = xdnn::search_seq2batch(ctx,
                               batch,
                               max_width,
                               cap_e_,
                               sentense.idx_sorted_32,
                               sentense.lod_32,
                               sentense.new_offset_32,
                               in,
                               seq2batch_out);
    CHECK_EQ(r, 0);

    r = xdnn::findmax<float>(ctx, in, cap_l * cap_e_, input_max_);
    CHECK_EQ(r, 0);
    fc_e2h0_.Infer(ctx, seq2batch_out, cap_l, fc_e2h_out, input_max_);
    fc_e2h1_.Infer(
        ctx, seq2batch_out, cap_l, fc_e2h_out + cap_l * cap_h_, input_max_);
    fc_e2h2_.Infer(
        ctx, seq2batch_out, cap_l, fc_e2h_out + cap_l * cap_h_ * 2, input_max_);
    r = xdnn::search_grnn<float, int16_t>(ctx,
                                          cap_l,
                                          cap_h_,
                                          cap_e_,
                                          max_width,
                                          sentense.new_offset_32,
                                          fc_e2h_out,
                                          dense_h2h_,
                                          gru_out,
                                          dense_h2h_max_[0],
                                          dense_h2h_max_[1],
                                          dense_h2h_max_[2]);
    CHECK_EQ(r, 0);

    r = xdnn::search_batch2seq(ctx,
                               batch,
                               max_width,
                               cap_h_,
                               sentense.idx_sorted_32,
                               sentense.lod_32,
                               sentense.new_offset_32,
                               gru_out,
                               out);
    CHECK_EQ(r, 0);
  }
};

class MMDNNAttentionOp {
  int dim_;
  float alpha0_;
  float alpha1_;
  MMDNNFcOp seqfc_;
  XPUScratchPadGuard hbm_buffer_guard_;
  float* hbm_buffer_{nullptr};
  // require: cap_l * dim_ + seqlen_square_sum
  // seqfc_out: [cap_l, dim_]
  // batchgemm0_out: [seqlen_square_sum]
  // seq_softmax_out: [seqlen_square_sum], reuse of batchgemm0_out
  // batchgemm1_out: [cap_l, dim_], reuse of seqfc_out

 public:
  void Init(lite::Tensor* att_fc_w,
            float att_fc_w_max,
            lite::Tensor* att_fc_b,
            int dim,
            int upper_bound_batch,
            int upper_bound_seqlen) {
    dim_ = dim;
    alpha0_ = 0.0883883461356163f;  // TODO(miaotianxiang):
    alpha1_ = 1.0f;

    seqfc_.Init(att_fc_w,
                att_fc_w_max,
                att_fc_b,
                dim_,
                dim_,
                xdnn::Activation_t::LINEAR);
    hbm_buffer_guard_ = TargetWrapperXPU::MallocScratchPad(
        (upper_bound_batch * (upper_bound_seqlen * dim_ +
                              upper_bound_seqlen * upper_bound_seqlen)) *
            sizeof(float),
        false);
    hbm_buffer_ = reinterpret_cast<float*>(hbm_buffer_guard_->addr_);
  }

  void Infer(xdnn::Context* ctx,
             const MMDNNIdInfo& sentense,
             const float* input,
             float* pool_out,
             float* l3_buffer = nullptr,
             int l3_size = 0) {
    int batch = sentense.batch;
    int cap_l = sentense.seqlen_sum;
    int max_width = sentense.seqlen_max;
    int* lod_32 = sentense.lod_32;

    float* seqfc_out = hbm_buffer_;
    float* batchgemm0_out = hbm_buffer_ + cap_l * dim_;
    float* seq_softmax_out = batchgemm0_out;
    float* batchgemm1_out = seqfc_out;
    if (l3_size > 0 &&
        l3_size >=
            (cap_l * dim_ + sentense.seqlen_square_sum) * sizeof(float)) {
      seqfc_out = l3_buffer;
      batchgemm0_out = l3_buffer + cap_l * dim_;
      seq_softmax_out = batchgemm0_out;
      batchgemm1_out = seqfc_out;
    }

    seqfc_.Infer(ctx, input, cap_l, seqfc_out);
    int r = 0;
    r = xdnn::search_noaligned_mat_mul(ctx,
                                       0,
                                       1,
                                       batch,
                                       lod_32,
                                       max_width,
                                       dim_,
                                       alpha0_,
                                       input,
                                       seqfc_out,
                                       batchgemm0_out);
    CHECK_EQ(r, 0);
    r = xdnn::search_seq_softmax(
        ctx, batchgemm0_out, seq_softmax_out, lod_32, batch, max_width);
    CHECK_EQ(r, 0);
    r = xdnn::search_noaligned_mat_mul(ctx,
                                       0,
                                       0,
                                       batch,
                                       lod_32,
                                       max_width,
                                       dim_,
                                       alpha1_,
                                       seq_softmax_out,
                                       input,
                                       batchgemm1_out);
    CHECK_EQ(r, 0);
    r = xdnn::sequence_pooling_forward(ctx,
                                       xdnn::Pooling_t::MAX_WITHOUT_INDEX,
                                       batch,
                                       lod_32,
                                       dim_,
                                       batchgemm1_out,
                                       nullptr,
                                       pool_out);
    CHECK_EQ(r, 0);
  }
};

class MMDNNMatchConvTopk {
  std::vector<int> topks_;
  int dim_t_;
  int dim_in_;
  int out_channel_;

  MMDNNFcOp xw_fc_;
  const int16_t* conv_weight_{nullptr};
  float conv_weight_max_;
  XPUScratchPadGuard hbm_buffer_guard_;
  float* hbm_buffer_{nullptr};
  // xw_out: [sum(left_len), dim_t_ * dim_in_]
  // xwy_out: [sum(left_len * right_len) * dim_t_]
  // conv_out: [sum(left_len * right_len) * out_channel_]
  // seq_concat_out: [sum(left_len * right_len) * (dim_t_ + out_channel_)]

  XPUScratchPadGuard left_lod_32_guard_;
  int* left_lod_32_{nullptr};
  XPUScratchPadGuard right_lod_32_guard_;
  int* right_lod_32_{nullptr};
  XPUScratchPadGuard match_lod_32_guard_;
  int* match_lod_32_{nullptr};
  XPUScratchPadGuard conv_lod_32_guard_;
  int* conv_lod_32_{nullptr};
  XPUScratchPadGuard topk_offset_32_guard_;
  int* topk_offset_32_{nullptr};
  XPUScratchPadGuard topks_xpu_guard_;
  int* topks_xpu_{nullptr};
  XPUScratchPadGuard useless_topk_pos_guard_;
  int* useless_topk_pos_{nullptr};

 public:
  float* seq_avg_topk_out{nullptr};

  void Init(lite::Tensor* input_w,
            float input_w_max,
            lite::Tensor* conv_w,
            float conv_w_max,
            int dim_t,
            int dim_in,
            int out_channel,
            int upper_bound_batch,
            int upper_bound_seqlen,
            const std::vector<int>& topks) {
    dim_t_ = dim_t;
    dim_in_ = dim_in;
    out_channel_ = out_channel;
    topks_ = topks;

    xw_fc_.Init(input_w,
                input_w_max,
                nullptr,
                dim_t_ * dim_in_,
                dim_in_,
                xdnn::Activation_t::LINEAR);
    conv_weight_ = conv_w->data<int16_t>();
    conv_weight_max_ = conv_w_max;

    hbm_buffer_guard_ = TargetWrapperXPU::MallocScratchPad(
        (upper_bound_batch * upper_bound_seqlen * dim_t_ * dim_in_ +
         upper_bound_batch * upper_bound_seqlen * upper_bound_seqlen *
             (dim_t_ + out_channel_) * 2) *
            sizeof(float),
        false);
    hbm_buffer_ = reinterpret_cast<float*>(hbm_buffer_guard_->addr_);

    left_lod_32_guard_ = TargetWrapperXPU::MallocScratchPad(
        (upper_bound_batch + 1) * sizeof(int), false);
    left_lod_32_ = reinterpret_cast<int*>(left_lod_32_guard_->addr_);
    right_lod_32_guard_ = TargetWrapperXPU::MallocScratchPad(
        (upper_bound_batch + 1) * sizeof(int), false);
    right_lod_32_ = reinterpret_cast<int*>(right_lod_32_guard_->addr_);
    match_lod_32_guard_ = TargetWrapperXPU::MallocScratchPad(
        (upper_bound_batch + 1) * sizeof(int), false);
    match_lod_32_ = reinterpret_cast<int*>(match_lod_32_guard_->addr_);
    conv_lod_32_guard_ = TargetWrapperXPU::MallocScratchPad(
        (upper_bound_batch + 1) * sizeof(int), false);
    conv_lod_32_ = reinterpret_cast<int*>(conv_lod_32_guard_->addr_);
    topk_offset_32_guard_ = TargetWrapperXPU::MallocScratchPad(
        (upper_bound_batch + 1) * sizeof(int), false);
    topk_offset_32_ = reinterpret_cast<int*>(topk_offset_32_guard_->addr_);
    topks_xpu_guard_ =
        TargetWrapperXPU::MallocScratchPad(topks_.size() * sizeof(int), false);
    topks_xpu_ = reinterpret_cast<int*>(topks_xpu_guard_->addr_);
    XPU_CALL(xpu_memcpy(topks_xpu_,
                        topks_.data(),
                        topks_.size() * sizeof(int),
                        XPUMemcpyKind::XPU_HOST_TO_DEVICE));
    useless_topk_pos_guard_ =
        TargetWrapperXPU::MallocScratchPad(4 * sizeof(int), false);
    useless_topk_pos_ = reinterpret_cast<int*>(useless_topk_pos_guard_->addr_);
  }

  void Infer(xdnn::Context* ctx,
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
    XPU_CALL(xpu_memcpy(left_lod_32_,
                        left_lod_32_cpu.data(),
                        left_lod_32_cpu.size() * sizeof(int),
                        XPUMemcpyKind::XPU_HOST_TO_DEVICE));
    std::vector<int> right_lod_32_cpu;
    for (auto e : right_lod) {
      right_lod_32_cpu.push_back(e);
    }
    XPU_CALL(xpu_memcpy(right_lod_32_,
                        right_lod_32_cpu.data(),
                        right_lod_32_cpu.size() * sizeof(int),
                        XPUMemcpyKind::XPU_HOST_TO_DEVICE));

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
      lod_match.push_back(lod_match.back() + imgsize * dim_t_);
      lod_conv.push_back(lod_conv.back() + imgsize * out_channel_);
      lod_topk.push_back(lod_topk.back() + imgsize * (dim_t_ + out_channel_));

      left_seqlen_max = std::max(left_seqlen_max, len_x);
      right_seqlen_max = std::max(right_seqlen_max, len_y);
      left_seqlen_sum += len_x;
      right_seqlen_sum += len_y;
    }
    XPU_CALL(xpu_memcpy(match_lod_32_,
                        lod_match.data(),
                        lod_match.size() * sizeof(int),
                        XPUMemcpyKind::XPU_HOST_TO_DEVICE));
    XPU_CALL(xpu_memcpy(conv_lod_32_,
                        lod_conv.data(),
                        lod_conv.size() * sizeof(int),
                        XPUMemcpyKind::XPU_HOST_TO_DEVICE));
    XPU_CALL(xpu_memcpy(topk_offset_32_,
                        lod_topk.data(),
                        lod_topk.size() * sizeof(int),
                        XPUMemcpyKind::XPU_HOST_TO_DEVICE));

    float* xwy_out = hbm_buffer_;
    float* conv_out = hbm_buffer_ + x_mul_y_sum * dim_t_;
    float* seq_concat_out = hbm_buffer_ + x_mul_y_sum * (dim_t_ + out_channel_);
    float* xw_out = hbm_buffer_ + x_mul_y_sum * (dim_t_ + out_channel_) * 2;
    int total_len = x_mul_y_sum * (dim_t_ + out_channel_) * 2 +
                    left_seqlen_sum * dim_t_ * dim_in_;
    if (l3_size > 0 && l3_size >= total_len * sizeof(float)) {
      xwy_out = l3_buffer;
      conv_out = l3_buffer + x_mul_y_sum * dim_t_;
      seq_concat_out = l3_buffer + x_mul_y_sum * (dim_t_ + out_channel_);
      xw_out = l3_buffer + x_mul_y_sum * (dim_t_ + out_channel_) * 2;
    }
    seq_avg_topk_out = out->mutable_data<float>(TARGET(kXPU));

    int max_width = std::max(left_seqlen_max, right_seqlen_max);
    xw_fc_.Infer(ctx, left->data<float>(), left_seqlen_sum, xw_out);
    int r = 0;
    r = xdnn::match_matrix_tensor(ctx,
                                  batch,
                                  xw_out,
                                  right->data<float>(),
                                  left_lod_32_,
                                  right_lod_32_,
                                  dim_t_,
                                  dim_in_,
                                  xwy_out,
                                  xw_fc_.out_max,
                                  xdnn::Activation_t::RELU,
                                  max_width);
    CHECK_EQ(r, 0);
    r = xdnn::search_varconv<float, int16_t>(
        ctx,
        batch,
        dim_t_,
        out_channel_,
        5,
        5,
        1,
        1,
        xwy_out,
        conv_weight_,
        right_lod_32_,
        left_lod_32_,
        conv_out,
        conv_weight_max_,
        xdnn::Activation_t::RELU);  // TODO(miaotianxiang):
    CHECK_EQ(r, 0);
    r = xdnn::sequence_concat(ctx,
                              xwy_out,
                              match_lod_32_,
                              conv_out,
                              conv_lod_32_,
                              seq_concat_out,
                              batch);
    CHECK_EQ(r, 0);
    r = xdnn::sequence_topk_avg_pooling(ctx,
                                        seq_concat_out,
                                        seq_avg_topk_out,
                                        useless_topk_pos_,
                                        batch,
                                        dim_t_ + out_channel_,
                                        topk_offset_32_,
                                        left_lod_32_,
                                        right_lod_32_,
                                        topks_xpu_,
                                        topks_.size());
    CHECK_EQ(r, 0);
  }
};

class MMDNNBidEmbGrnnAtt {
  const float* table_{nullptr};
  int table_len_;
  int emb_dim_;
  int cap_h_;
  MMDNNGrnnOp bi_fw_;
  MMDNNGrnnOp bi_rv_;
  MMDNNAttentionOp att_;
  XPUScratchPadGuard hbm_buffer_guard_;
  float* hbm_buffer_{nullptr};
  // require at least: 4 * cap_l * emb_dim_
  // emb_rv: [cap_l, emb_dim_]
  // grnn_fw: [cap_l, emb_dim_]
  // grnn_rv: [cap_l, emb_dim_]
  // grnn_rv_rv: [cap_l, emb_dim_]
  // concat_2in: [cap_l, 2 * emb_dim_]
  // L3.bi_fw: 5 * cap_l * emb_dim_
  // L3.bi_rv: 5 * cap_l * emb_dim_
  // L3.att:   cap_l * 2 * emb_dim_ + seqlen_square_sum

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
  float* emb_fw{nullptr};
  float* concat_3in{nullptr};
  float* pool_fw{nullptr};
  float* pool_rv{nullptr};
  float* att_out{nullptr};

  void Init(lite::Tensor* table,
            lite::Tensor* fw_wh,
            const std::vector<float>& fw_wh_maxs,
            lite::Tensor* fw_wi,
            const std::vector<float>& fw_wi_maxs,
            lite::Tensor* rv_wh,
            const std::vector<float>& rv_wh_maxs,
            lite::Tensor* rv_wi,
            const std::vector<float>& rv_wi_maxs,
            lite::Tensor* att_fc_w,
            float att_fc_w_max,
            lite::Tensor* att_fc_b,
            int upper_bound_batch,
            int upper_bound_seqlen) {
    table_ = table->data<float>();
    table_len_ = table->dims()[0];
    emb_dim_ = table->dims()[1];
    cap_h_ = emb_dim_;
    int max_cap_l = upper_bound_batch * upper_bound_seqlen;

    bi_fw_.Init(
        fw_wh, fw_wh_maxs, fw_wi, fw_wi_maxs, emb_dim_, cap_h_, max_cap_l);
    bi_rv_.Init(
        rv_wh, rv_wh_maxs, rv_wi, rv_wi_maxs, emb_dim_, cap_h_, max_cap_l);
    att_.Init(att_fc_w,
              att_fc_w_max,
              att_fc_b,
              2 * cap_h_,
              upper_bound_batch,
              upper_bound_seqlen);

    hbm_buffer_guard_ = TargetWrapperXPU::MallocScratchPad(
        4 * max_cap_l * cap_h_ * sizeof(float), false);
    hbm_buffer_ = reinterpret_cast<float*>(hbm_buffer_guard_->addr_);
  }

  void Infer(xdnn::Context* ctx,
             int batch,
             const MMDNNIdInfo& sentense,
             lite::Tensor* grnn_fw_pool_out,
             lite::Tensor* grnn_rv_pool_out,
             lite::Tensor* att_pool_out,
             lite::Tensor* concat_3in1_out,
             lite::Tensor* emb_fw_out,
             float* l3_buffer = nullptr,
             int l3_size = 0) {
    int cap_l = sentense.seqlen_sum;
    int slot_len = cap_l * cap_h_;

    float* emb_rv = hbm_buffer_;
    float* grnn_fw = hbm_buffer_ + slot_len;
    float* grnn_rv = hbm_buffer_ + slot_len;
    float* grnn_rv_rv = hbm_buffer_;
    float* concat_2in = hbm_buffer_ + 2 * slot_len;
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

    int r = 0;
    r = xdnn::search_bid_emb_ew(ctx,
                                batch,
                                sentense.lod_64,
                                sentense.id0_64,
                                sentense.id1_64,
                                table_,
                                table_len_,
                                emb_dim_,
                                emb_fw,
                                emb_rv,
                                table_len_ - 2,
                                1);
    CHECK_EQ(r, 0);
    bi_rv_.Infer(ctx,
                 sentense,
                 emb_rv,
                 grnn_rv,
                 l3_buffer + 2 * slot_len,
                 l3_size - 2 * slot_len * sizeof(float));
    r = xdnn::sequence_reverse(
        ctx, batch, sentense.lod_32, cap_h_, grnn_rv, grnn_rv_rv);
    CHECK_EQ(r, 0);
    r = xdnn::sequence_pooling_forward(ctx,
                                       xdnn::Pooling_t::LAST,
                                       batch,
                                       sentense.lod_32,
                                       cap_h_,
                                       grnn_rv,
                                       nullptr,
                                       pool_rv);
    CHECK_EQ(r, 0);

    bi_fw_.Infer(ctx,
                 sentense,
                 emb_fw,
                 grnn_fw,
                 l3_buffer + 2 * slot_len,
                 l3_size - 2 * slot_len * sizeof(float));
    r = xdnn::sequence_pooling_forward(ctx,
                                       xdnn::Pooling_t::LAST,
                                       batch,
                                       sentense.lod_32,
                                       cap_h_,
                                       grnn_fw,
                                       nullptr,
                                       pool_fw);
    CHECK_EQ(r, 0);
    const int concat_widths[] = {cap_h_, cap_h_, cap_h_};
    const float* concat_ptrs[] = {emb_fw, grnn_fw, grnn_rv_rv};
    r = xdnn::concat<float>(
        ctx, cap_l, concat_widths + 1, 2, concat_ptrs + 1, concat_2in);
    CHECK_EQ(r, 0);
    r = xdnn::concat<float>(
        ctx, cap_l, concat_widths, 3, concat_ptrs, concat_3in);
    CHECK_EQ(r, 0);
    att_.Infer(ctx,
               sentense,
               concat_2in,
               att_out,
               l3_buffer + 4 * slot_len,
               l3_size - 4 * slot_len * sizeof(float));
  }
};

class MMDNNEmbAtt {
  const float* table_{nullptr};
  int table_len_;
  int emb_dim_;
  MMDNNAttentionOp att_;

 public:
  float* emb_fw{nullptr};
  float* att_out{nullptr};

  void Init(lite::Tensor* table,
            lite::Tensor* att_fc_w,
            float att_fc_w_max,
            lite::Tensor* att_fc_b,
            int upper_bound_batch,
            int upper_bound_seqlen) {
    table_ = table->data<float>();
    table_len_ = table->dims()[0];
    emb_dim_ = table->dims()[1];
    att_.Init(att_fc_w,
              att_fc_w_max,
              att_fc_b,
              emb_dim_,
              upper_bound_batch,
              upper_bound_seqlen);
  }

  void Infer(xdnn::Context* ctx,
             int batch,
             const MMDNNIdInfo& sentense,
             lite::Tensor* att_pool_out,
             lite::Tensor* emb_fw_out,
             float* l3_buffer = nullptr,
             int l3_size = 0) {
    emb_fw = emb_fw_out->mutable_data<float>(TARGET(kXPU));
    att_out = att_pool_out->mutable_data<float>(TARGET(kXPU));

    int cap_l = sentense.lod.back();
    const float* emb_tables[] = {table_, table_};
    const int64_t* emb_indices[] = {sentense.id0_64, sentense.id1_64};
    int r =
        xdnn::embedding_with_ewadd<float, int64_t, false, false>(ctx,
                                                                 emb_dim_,
                                                                 cap_l,
                                                                 2,
                                                                 table_len_ - 2,
                                                                 emb_tables,
                                                                 emb_indices,
                                                                 nullptr,
                                                                 nullptr,
                                                                 emb_fw);
    CHECK_EQ(r, 0);
    att_.Infer(ctx, sentense, emb_fw, att_out, l3_buffer, l3_size);
  }
};

class MMDNNMergeAll {
  MMDNNGrnnOp coverage_fw_;
  MMDNNGrnnOp coverage_rv_;
  int cap_e_;
  int cap_h_;

  // TODO(miaotianxiang):
  const int fc0_k_ = 1152;
  const int fc0_n_ = 512;
  const int fc1_k_ = 640;
  const int fc1_n_ = 320;
  const int fc2_k_ = 320;
  const int fc2_n_ = 1;
  MMDNNFcOp fc0_;
  MMDNNFcOp fc1_;
  MMDNNFcOp fc2_;

  XPUScratchPadGuard hbm_buffer_guard_;
  float* hbm_buffer_{nullptr};
  // topk_concat_out_fw:  [cap_l, cap_e_] <= [cap_l, cap_h_]
  // topk_concat_out_rv:  [cap_l, cap_e_] <= [cap_l, cap_h_]
  // grnn_fw:             [cap_l, cap_h_]
  // grnn_rv:             [cap_l, cap_h_]
  // pool_fw:             [batch, cap_h_]
  // pool_rv:             [batch, cap_h_]
  // fc0_in:              [batch, fc0_k_]
  // fc0_out:             [batch, fc0_n_]
  // fc1_in:              [batch, fc1_k_]
  // fc1_out:             [batch, fc1_n_]
  // fc2_out:             [batch, fc2_n_]

 public:
  void Init(lite::Tensor* grnn_fw_wh,
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
            int upper_bound_batch,
            int upper_bound_seqlen) {
    int max_cap_l = upper_bound_batch * upper_bound_seqlen;
    cap_e_ = grnn_fw_wi->dims()[2];
    cap_h_ = grnn_fw_wi->dims()[1];

    coverage_fw_.Init(grnn_fw_wh,
                      grnn_fw_wh_maxs,
                      grnn_fw_wi,
                      grnn_fw_wi_maxs,
                      cap_e_,
                      cap_h_,
                      max_cap_l);
    coverage_rv_.Init(grnn_rv_wh,
                      grnn_rv_wh_maxs,
                      grnn_rv_wi,
                      grnn_rv_wi_maxs,
                      cap_e_,
                      cap_h_,
                      max_cap_l);

    fc0_.Init(
        fc0_w, fc0_w_max, fc0_b, fc0_n_, fc0_k_, xdnn::Activation_t::RELU);
    fc1_.Init(
        fc1_w, fc1_w_max, fc1_b, fc1_n_, fc1_k_, xdnn::Activation_t::RELU);
    fc2_.Init(
        fc2_w, fc2_w_max, fc2_b, fc2_n_, fc2_k_, xdnn::Activation_t::LINEAR);

    int hbm_total_len = max_cap_l * cap_e_ * 2 + max_cap_l * cap_h_ * 2 +
                        upper_bound_batch * (2 * cap_h_ + fc0_k_ + fc0_n_ +
                                             fc1_k_ + fc1_n_ + fc2_n_);
    hbm_buffer_guard_ = TargetWrapperXPU::MallocScratchPad(
        hbm_total_len * sizeof(float), false);
    hbm_buffer_ = reinterpret_cast<float*>(hbm_buffer_guard_->addr_);
  }

  void Infer(xdnn::Context* ctx,
             const MMDNNIdInfo& sentense,
             const std::vector<lite::Tensor*> concat_topk_x,
             const std::vector<lite::Tensor*> concat_7in1_x,
             lite::Tensor* out,
             float* l3_buffer = nullptr,
             int l3_size = 0) {
    int batch = sentense.batch;
    int cap_l = sentense.seqlen_sum;

    float* topk_concat_out_fw = hbm_buffer_;
    int hbm_total_len =
        cap_l * cap_e_ * 2 + cap_l * cap_h_ * 2 +
        batch * (2 * cap_h_ + fc0_k_ + fc0_n_ + fc1_k_ + fc1_n_ + fc2_n_);
    if (l3_size > 0 && l3_size >= hbm_total_len * sizeof(float)) {
      topk_concat_out_fw = l3_buffer;
    }
    float* topk_concat_out_rv = topk_concat_out_fw + cap_l * cap_e_;
    float* grnn_fw = topk_concat_out_rv + cap_l * cap_e_;
    float* grnn_rv = grnn_fw + cap_l * cap_h_;
    float* pool_fw = grnn_rv + cap_l * cap_h_;
    float* pool_rv = pool_fw + batch * cap_h_;
    float* fc0_in = pool_fw + batch * cap_h_ * 2;
    float* fc0_out = fc0_in + batch * fc0_k_;
    float* fc1_in = fc0_out + batch * fc0_n_;
    float* fc1_out = fc1_in + batch * fc1_k_;
    // float* fc2_out = fc1_out + batch * fc1_n_;
    float* fc2_out = out->mutable_data<float>(TARGET(kXPU));

    std::vector<int> concat_widths;
    std::vector<const float*> concat_ptrs;
    for (const auto* t : concat_topk_x) {
      concat_widths.push_back(static_cast<int>(t->dims()[1]));
      concat_ptrs.push_back(t->data<float>());
    }
    int r = 0;
    r = xdnn::concat<float>(ctx,
                            cap_l,
                            concat_widths.data(),
                            concat_widths.size(),
                            concat_ptrs.data(),
                            topk_concat_out_fw);
    CHECK_EQ(r, 0);
    r = xdnn::sequence_reverse(ctx,
                               batch,
                               sentense.lod_32,
                               cap_e_,
                               topk_concat_out_fw,
                               topk_concat_out_rv);
    CHECK_EQ(r, 0);
    coverage_fw_.Infer(ctx,
                       sentense,
                       topk_concat_out_fw,
                       grnn_fw,
                       l3_buffer + hbm_total_len,
                       l3_size - hbm_total_len * sizeof(float));
    coverage_rv_.Infer(ctx,
                       sentense,
                       topk_concat_out_rv,
                       grnn_rv,
                       l3_buffer + hbm_total_len,
                       l3_size - hbm_total_len * sizeof(float));
    r = xdnn::sequence_pooling_forward(ctx,
                                       xdnn::Pooling_t::LAST,
                                       batch,
                                       sentense.lod_32,
                                       cap_h_,
                                       grnn_fw,
                                       nullptr,
                                       pool_fw);
    CHECK_EQ(r, 0);
    r = xdnn::sequence_pooling_forward(ctx,
                                       xdnn::Pooling_t::LAST,
                                       batch,
                                       sentense.lod_32,
                                       cap_h_,
                                       grnn_rv,
                                       nullptr,
                                       pool_rv);
    CHECK_EQ(r, 0);

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
    const int concat_widths_fc1[] = {cap_h_, cap_h_, fc0_n_};
    const float* concat_ptrs_fc1[] = {pool_fw, pool_rv, fc0_out};

    r = xdnn::concat<float>(
        ctx, batch, concat_widths_fc0, 7, concat_ptrs_fc0, fc0_in);
    CHECK_EQ(r, 0);
    fc0_.Infer(ctx, fc0_in, batch, fc0_out);
    r = xdnn::concat<float>(
        ctx, batch, concat_widths_fc1, 3, concat_ptrs_fc1, fc1_in);
    CHECK_EQ(r, 0);
    fc1_.Infer(ctx, fc1_in, batch, fc1_out);
    fc2_.Infer(ctx, fc1_out, batch, fc2_out);
  }
};

class XPUMmdnnBidEmbGrnnAttCompute
    : public KernelLite<TARGET(kXPU), PRECISION(kFloat)> {
 public:
  using param_t = operators::XPUMmdnnBidEmbGrnnAttParam;

  void PrepareForRun() override;

  void Run() override;

 private:
  MMDNNIdInfo id_;
  MMDNNBidEmbGrnnAtt compound_;
};

void XPUMmdnnBidEmbGrnnAttCompute::PrepareForRun() {
  auto& param = this->Param<param_t>();

  id_.Init(XPU_MAX_LOD_SIZE, XPU_MAX_LOD_SEQ_LEN);
  compound_.Init(param.emb_tbl,
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
                 XPU_MAX_LOD_SIZE,
                 XPU_MAX_LOD_SEQ_LEN);
}

void XPUMmdnnBidEmbGrnnAttCompute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->As<XPUContext>();

  auto* xpu_ctx = ctx.GetRawContext();

  int batch = param.id0->lod()[0].size() - 1;
  id_.Update(param.id0, param.id1);
  compound_.Infer(ctx.GetRawContext(),
                  batch,
                  id_,
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

class XPUMmdnnBidEmbGrnnAttCompute2
    : public KernelLite<TARGET(kXPU), PRECISION(kFloat)> {
 public:
  using param_t = operators::XPUMmdnnBidEmbGrnnAttParam2;

  void PrepareForRun() override;

  void Run() override;

 private:
  MMDNNIdInfo id_;
  MMDNNBidEmbGrnnAtt compound_;
};

void XPUMmdnnBidEmbGrnnAttCompute2::PrepareForRun() {
  auto& param = this->Param<param_t>();

  id_.Init(XPU_MAX_LOD_SIZE, XPU_MAX_LOD_SEQ_LEN);
  compound_.Init(param.emb_tbl,
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
                 XPU_MAX_LOD_SIZE,
                 XPU_MAX_LOD_SEQ_LEN);
}

void XPUMmdnnBidEmbGrnnAttCompute2::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->As<XPUContext>();

  auto* xpu_ctx = ctx.GetRawContext();

  int batch = param.id0->lod()[0].size() - 1;
  id_.Update(param.id0, param.id1);
  compound_.Infer(ctx.GetRawContext(),
                  batch,
                  id_,
                  param.grnn_fw_pool_out,
                  param.grnn_rv_pool_out,
                  param.att_pool_out,
                  param.concat_3in1_out,
                  param.emb_fw_out,
                  reinterpret_cast<float*>(
                      reinterpret_cast<char*>(xpu_ctx->workspace_l3_ptr) +
                      xpu_ctx->used_l3_size),
                  xpu_ctx->workspace_l3_size - xpu_ctx->used_l3_size);

  int num = param.id0->numel();
  int embed_dim = param.emb_tbl->dims()[1];

  // TODO(miaotianxiang):
  int r = xdnn::embedding<float, int64_t>(
      ctx.GetRawContext(),                               /* context */
      num,                                               /* num */
      param.id0->data<int64_t>(),                        /* indices */
      embed_dim,                                         /* embed_dim */
      param.emb_tbl->data<float>(),                      /* table */
      param.emb0_out->mutable_data<float>(TARGET(kXPU)), /* top */
      128000 /* padding_idx */);
  CHECK_EQ(r, 0);
}

class XPUMmdnnBidEmbAttCompute
    : public KernelLite<TARGET(kXPU), PRECISION(kFloat)> {
 public:
  using param_t = operators::XPUMmdnnBidEmbAttParam;

  void PrepareForRun() override;

  void Run() override;

 private:
  MMDNNIdInfo id_;
  MMDNNEmbAtt compound_;
};

void XPUMmdnnBidEmbAttCompute::PrepareForRun() {
  auto& param = this->Param<param_t>();

  id_.Init(XPU_MAX_LOD_SIZE, XPU_MAX_LOD_SEQ_LEN);
  compound_.Init(param.emb_tbl,
                 param.att_fc_w,
                 param.att_fc_w_max,
                 param.att_fc_b,
                 XPU_MAX_LOD_SIZE,
                 XPU_MAX_LOD_SEQ_LEN);
}

void XPUMmdnnBidEmbAttCompute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->As<XPUContext>();

  auto* xpu_ctx = ctx.GetRawContext();

  int batch = param.id0->lod()[0].size() - 1;
  id_.Update(param.id0, param.id1);
  compound_.Infer(ctx.GetRawContext(),
                  batch,
                  id_,
                  param.att_pool_out,
                  param.emb_fw_out,
                  reinterpret_cast<float*>(
                      reinterpret_cast<char*>(xpu_ctx->workspace_l3_ptr) +
                      xpu_ctx->used_l3_size),
                  xpu_ctx->workspace_l3_size - xpu_ctx->used_l3_size);
}

class XPUMmdnnMatchConvTopkCompute
    : public KernelLite<TARGET(kXPU), PRECISION(kFloat)> {
 public:
  using param_t = operators::XPUMmdnnMatchConvTopkParam;

  void PrepareForRun() override;

  void Run() override;

 private:
  MMDNNMatchConvTopk compound_;
};

void XPUMmdnnMatchConvTopkCompute::PrepareForRun() {
  auto& param = this->Param<param_t>();

  compound_.Init(param.input_w,
                 param.input_w_max,
                 param.conv_w,
                 param.conv_w_max,
                 param.dim_t,
                 param.input_w->dims()[0],
                 param.output_channel,
                 XPU_MAX_LOD_SIZE,
                 XPU_MAX_LOD_SEQ_LEN,
                 param.topks);
}

void XPUMmdnnMatchConvTopkCompute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->As<XPUContext>();

  auto* xpu_ctx = ctx.GetRawContext();

  compound_.Infer(ctx.GetRawContext(),
                  param.input_x,
                  param.input_y,
                  param.topk_out,
                  reinterpret_cast<float*>(
                      reinterpret_cast<char*>(xpu_ctx->workspace_l3_ptr) +
                      xpu_ctx->used_l3_size),
                  xpu_ctx->workspace_l3_size - xpu_ctx->used_l3_size);
}

class XPUMmdnnMergeAllCompute
    : public KernelLite<TARGET(kXPU), PRECISION(kFloat)> {
 public:
  using param_t = operators::XPUMmdnnMergeAllParam;

  void PrepareForRun() override;

  void Run() override;

 private:
  MMDNNIdInfo id_;
  MMDNNMergeAll compound_;
};

void XPUMmdnnMergeAllCompute::PrepareForRun() {
  auto& param = this->Param<param_t>();

  id_.Init(XPU_MAX_LOD_SIZE, XPU_MAX_LOD_SEQ_LEN);
  compound_.Init(param.grnn_fw_wh,
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
                 XPU_MAX_LOD_SIZE,
                 XPU_MAX_LOD_SEQ_LEN);
}

void XPUMmdnnMergeAllCompute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->As<XPUContext>();

  auto* xpu_ctx = ctx.GetRawContext();

  id_.Update(param.concat_topk_x[0], param.concat_topk_x[1]);
  compound_.Infer(ctx.GetRawContext(),
                  id_,
                  param.concat_topk_x,
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
                     paddle::lite::kernels::xpu::XPUMmdnnBidEmbGrnnAttCompute,
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

REGISTER_LITE_KERNEL(__xpu__mmdnn_bid_emb_grnn_att2,
                     kXPU,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::xpu::XPUMmdnnBidEmbGrnnAttCompute2,
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
    .BindOutput("emb0_out", {LiteType::GetTensorTy(TARGET(kXPU))})
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
                     paddle::lite::kernels::xpu::XPUMmdnnBidEmbAttCompute,
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
                     paddle::lite::kernels::xpu::XPUMmdnnMatchConvTopkCompute,
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
                     paddle::lite::kernels::xpu::XPUMmdnnMergeAllCompute,
                     def)
    .BindInput("concat_7in1_x", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("concat_topk_x", {LiteType::GetTensorTy(TARGET(kXPU))})
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
