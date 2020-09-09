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

#include <gtest/gtest.h>
#include "lite/api/paddle_use_kernels.h"
#include "lite/api/paddle_use_ops.h"
#include "lite/core/arena/framework.h"
#include "lite/tests/utils/fill_data.h"

namespace paddle {
namespace lite {

class PoolComputeTest : public arena::TestCase {
 protected:
  // common attributes for this op.
  std::string op_type_ = "pool2d";
  std::string x_ = "x";
  std::string out_ = "out";
  DDim dims_{{1, 2, 3, 4}};
  std::string pooling_type_ = "max";
  bool global_pooling_ = false;
  std::vector<int> strides_{1, 1};
  std::vector<int> paddings_{0, 0};
  std::vector<int> ksize_{2, 2};
  bool exclusive_ = true;
  bool ceil_mode_ = false;
  bool adaptive_ = false;
  std::string padding_algorithm_;

 public:
  PoolComputeTest(const Place& place,
                  const std::string& alias,
                  DDim dims,
                  std::string pooling_type,
                  bool global_pooling,
                  std::vector<int> strides = {1, 1},
                  std::vector<int> paddings = {0, 0},
                  std::vector<int> ksize = {2, 2},
                  bool exclusive = true,
                  bool ceil_mode = false,
                  bool adaptive = false,
                  std::string padding_algorithm = "")
      : TestCase(place, alias),
        dims_(dims),
        pooling_type_(pooling_type),
        global_pooling_(global_pooling),
        strides_(strides),
        paddings_(paddings),
        ksize_(ksize),
        exclusive_(exclusive),
        ceil_mode_(ceil_mode),
        adaptive_(adaptive) {}

  void RunBaseline(Scope* scope) override {
    std::vector<int> paddings_new{paddings_};
    if (paddings_new.size() == 1L) {
      paddings_new = std::vector<int>(4, paddings_new[0]);
    } else if (paddings_new.size() == 2L) {
      paddings_new.insert(paddings_new.begin(), paddings_new[0]);
      paddings_new.insert(paddings_new.begin() + 2, paddings_new[2]);
    }
    CHECK_EQ(paddings_new.size(), 4L);
    if (padding_algorithm_ == "SAME") {
      for (int i = 0; i < strides_.size(); ++i) {
        int out_size = (dims_[i + 2] + strides_[i] - 1) / strides_[i];
        int pad_sum =
            std::max((out_size - 1) * strides_[i] + ksize_[i] - dims_[i + 2],
                     (int64_t)0);
        int pad_0 = pad_sum / 2;
        int pad_1 = pad_sum - pad_0;
        *(paddings_new.begin() + i * 2) = pad_0;
        *(paddings_new.begin() + i * 2 + 1) = pad_1;
      }
    }
    if (padding_algorithm_ == "VALID" || global_pooling_ || adaptive_) {
      for (size_t i = 0; i < paddings_new.size(); i++) {
        paddings_new[i] = 0;
      }
    }

    std::vector<int> ksize_new{ksize_};
    if (global_pooling_) {
      ksize_new.clear();
      ksize_new.push_back(dims_[2]);
      ksize_new.push_back(dims_[3]);
    }

    std::vector<int64_t> out_shape{dims_[0], dims_[1]};
    if (adaptive_) {
      out_shape.insert(out_shape.end(), ksize_new.begin(), ksize_new.end());
    } else {
      for (size_t i = 0; i < ksize_new.size(); ++i) {
        int out_size;
        if (!ceil_mode_) {
          out_size = (dims_[i + 2] - ksize_new[i] + paddings_new[2 * i] +
                      paddings_new[2 * i + 1]) /
                         strides_[i] +
                     1;
        } else {
          out_size = (dims_[i + 2] - ksize_new[i] + paddings_new[2 * i] +
                      paddings_new[2 * i + 1] + strides_[i] - 1) /
                         strides_[i] +
                     1;
        }
        out_shape.push_back(out_size);
      }
    }

    auto out = scope->NewTensor(out_);
    CHECK(out);
    out->Resize(DDim(out_shape));
    auto out_dims = out->dims();
    auto dst_ptr = out->mutable_data<float>();

    auto x = scope->FindTensor(x_);
    auto src_ptr = x->data<float>();

    int in_n = dims_[0];
    int in_c = dims_[1];
    int in_h = dims_[2];
    int in_w = dims_[3];
    int size_in_n = in_c * in_h * in_w;
    int size_in_c = in_h * in_w;

    int out_h = out_dims[2];
    int out_w = out_dims[3];
    int size_out_n = in_c * out_h * out_w;
    int size_out_c = out_h * out_w;

    int window_h = ksize_new[0];
    int window_w = ksize_new[1];
    int stride_h = strides_[0];
    int stride_w = strides_[1];
    int pad_t = paddings_new[0];
    int pad_l = paddings_new[2];

    if (global_pooling_) {
      for (int n = 0; n < in_n; ++n) {
        for (int c = 0; c < in_c; ++c) {
          const float* src = src_ptr + n * size_in_n + c * size_in_c;
          float res = src[0];
          if (pooling_type_ == "max") {
            for (int i = 1; i < size_in_c; ++i) {
              float cur_val = src[i];
              res = cur_val > res ? cur_val : res;
            }
          } else if (pooling_type_ == "avg") {
            for (int i = 1; i < size_in_c; ++i) {
              float cur_val = src[i];
              res += cur_val;
            }
            res /= size_in_c;
          }
          dst_ptr[n * size_out_n + c] = res;
        }
      }
    } else {
      for (int n = 0; n < in_n; ++n) {
        for (int c = 0; c < in_c; ++c) {
          for (int h = 0; h < out_h; ++h) {
            int sh = h * stride_h;
            int eh = sh + window_h;
            sh = (sh - pad_t) < 0 ? 0 : sh - pad_t;
            eh = (eh - pad_t) > in_h ? in_h : eh - pad_t;
            for (int w = 0; w < out_w; ++w) {
              int sw = w * stride_w;
              int ew = sw + window_w;
              sw = (sw - pad_l) < 0 ? 0 : sw - pad_l;
              ew = (ew - pad_l) > in_w ? in_w : ew - pad_l;
              int pooling_size = (ew - sw) * (eh - sh);
              if (pooling_size == 0) continue;
              float res = 0.f;
              for (int kh = sh; kh < eh; ++kh) {
                for (int kw = sw; kw < ew; ++kw) {
                  int src_idx = n * size_in_n + c * size_in_c + kh * in_w + kw;
                  if (kh == sh && kw == sw) {
                    res = src_ptr[src_idx];
                  } else {
                    if (pooling_type_ == "max") {
                      res = res >= src_ptr[src_idx] ? res : src_ptr[src_idx];
                    }
                    if (pooling_type_ == "avg") {
                      res += src_ptr[src_idx];
                    }
                  }
                }
              }
              if (pooling_type_ == "avg") {
                if (exclusive_) {
                  res /= pooling_size;
                } else {
                  res /= window_h * window_w;
                }
              }
              dst_ptr[n * size_out_n + c * size_out_c + h * out_w + w] = res;
            }
          }
        }
      }
    }
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType(op_type_);
    op_desc->SetInput("X", {x_});
    op_desc->SetOutput("Out", {out_});
    op_desc->SetAttr("pooling_type", pooling_type_);
    op_desc->SetAttr("global_pooling", global_pooling_);
    op_desc->SetAttr("strides", strides_);
    op_desc->SetAttr("paddings", paddings_);
    op_desc->SetAttr("ksize", ksize_);
    op_desc->SetAttr("exclusive", exclusive_);
    op_desc->SetAttr("ceil_mode", ceil_mode_);
    op_desc->SetAttr("adaptive", adaptive_);
    if (!padding_algorithm_.empty()) {
      op_desc->SetAttr("padding_algorithm", padding_algorithm_);
    }
  }

  void PrepareData() override {
    std::vector<float> din(dims_.production());
    fill_data_rand(din.data(), -1.f, 1.f, dims_.production());
    SetCommonTensor(x_, dims_, din.data());
  }
};

void TestPoolGlobal(Place place, float abs_error = 2e-5) {
  for (auto dims : std::vector<std::vector<int64_t>>{{2, 3, 4, 5}}) {
    for (std::string pooling_type : {"max", "avg"}) {
      std::unique_ptr<arena::TestCase> tester(
          new PoolComputeTest(place, "def", DDim(dims), pooling_type, true));
      arena::Arena arena(std::move(tester), place, abs_error);
      arena.TestPrecision();
    }
  }
}

void TestPoolAlgorithm(Place place, float abs_error = 2e-5) {
  for (auto dims : std::vector<std::vector<int64_t>>{{2, 3, 4, 5}}) {
    for (auto pooling_type : {"max", "avg"}) {
      for (auto padding_algorithm : {"SAME", "VALID"}) {
        std::unique_ptr<arena::TestCase> tester(
            new PoolComputeTest(place,
                                "def",
                                DDim(dims),
                                pooling_type,
                                false,
                                {2, 2},
                                {0, 0},
                                {2, 2},
                                true,
                                false,
                                false,
                                padding_algorithm));
        arena::Arena arena(std::move(tester), place, abs_error);
        arena.TestPrecision();
      }
    }
  }
}

void TestPoolHelper(Place place,
                    float abs_error,
                    std::vector<int64_t> dims,
                    std::string pooling_type,
                    std::vector<int> strides,
                    std::vector<int> paddings,
                    std::vector<int> ksize,
                    bool exclusive = true,
                    bool ceil_mode = false,
                    bool adaptive = false,
                    std::string padding_algorithm = "") {
  std::unique_ptr<arena::TestCase> tester(
      new PoolComputeTest(place,
                          "def",
                          DDim(dims),
                          pooling_type,
                          false,
                          strides,
                          paddings,
                          ksize,
                          exclusive,
                          ceil_mode,
                          adaptive,
                          padding_algorithm));
  arena::Arena arena(std::move(tester), place, abs_error);
  arena.TestPrecision();
}

void TestPoolStrides(Place place, float abs_error = 2e-5) {
  for (auto pooling_type : {"max", "avg"}) {
    TestPoolHelper(
        place, abs_error, {2, 3, 6, 7}, pooling_type, {1, 1}, {0, 0}, {2, 2});
    TestPoolHelper(
        place, abs_error, {2, 3, 6, 7}, pooling_type, {1, 2}, {0, 0}, {2, 2});
    TestPoolHelper(
        place, abs_error, {2, 3, 6, 7}, pooling_type, {2, 2}, {0, 0}, {2, 2});
  }
}

void TestPoolPaddings(Place place, float abs_error = 2e-5) {
  for (auto pooling_type : {"max", "avg"}) {
    TestPoolHelper(
        place, abs_error, {2, 3, 6, 7}, pooling_type, {1, 1}, {0, 0}, {2, 2});
#if !defined(LITE_WITH_XPU)
    TestPoolHelper(
        place, abs_error, {2, 3, 6, 7}, pooling_type, {1, 1}, {1, 1}, {2, 2});
    TestPoolHelper(place,
                   abs_error,
                   {2, 3, 6, 7},
                   pooling_type,
                   {1, 1},
                   {0, 0, 1, 1},
                   {2, 2});
    TestPoolHelper(place,
                   abs_error,
                   {2, 3, 6, 7},
                   pooling_type,
                   {1, 1},
                   {1, 0, 1, 0},
                   {2, 2});
    TestPoolHelper(place,
                   abs_error,
                   {2, 3, 6, 7},
                   pooling_type,
                   {1, 1},
                   {1, 0, 0, 1},
                   {2, 2});
#endif
  }
}

void TestPoolKsize(Place place, float abs_error = 2e-5) {
  for (auto pooling_type : {"max", "avg"}) {
    for (auto ksize : {2, 3}) {
      TestPoolHelper(place,
                     abs_error,
                     {2, 3, 6, 7},
                     pooling_type,
                     {1, 1},
                     {0, 0},
                     {ksize, ksize});
#if !defined(LITE_WITH_XPU)
      TestPoolHelper(place,
                     abs_error,
                     {2, 3, 6, 7},
                     pooling_type,
                     {2, 2},
                     {1, 1},
                     {ksize, ksize});
#endif
    }
  }
}

void TestPoolCeilMode(Place place, float abs_error = 2e-5) {
  for (auto pooling_type : {"max", "avg"}) {
#if defined(LITE_WITH_XPU)
    if (pooling_type == std::string("max")) continue;
#endif
    TestPoolHelper(place,
                   abs_error,
                   {2, 3, 6, 6},
                   pooling_type,
                   {2, 2},
                   {0, 0, 0, 0},
                   {3, 3},
                   true,
                   true);
  }
}

TEST(Pool, precision) {
  LOG(INFO) << "test pool op";
  float abs_error = 2e-5;
  Place place;
#if defined(LITE_WITH_NPU)
  place = TARGET(kNPU);
  abs_error = 1e-2;  // Using fp16 in NPU
#elif defined(LITE_WITH_HUAWEI_ASCEND_NPU)
  place = TARGET(kHuaweiAscendNPU);
  abs_error = 1e-2;  // precision_mode default is force_fp16
#elif defined(LITE_WITH_XPU) && defined(LITE_WITH_XTCL)  // NOLINT
  place = TARGET(kXPU);
#else
  return;
#endif

  TestPoolGlobal(place, abs_error);
  TestPoolAlgorithm(place, abs_error);
  TestPoolStrides(place, abs_error);
  TestPoolPaddings(place, abs_error);
  TestPoolKsize(place, abs_error);
  TestPoolCeilMode(place, abs_error);
}

}  // namespace lite
}  // namespace paddle
