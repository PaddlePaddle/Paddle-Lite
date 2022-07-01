// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
#include "lite/core/test/arena/framework.h"
#include "lite/tests/utils/fill_data.h"

namespace paddle {
namespace lite {

class AdaptivePoolTester : public arena::TestCase {
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
  bool adaptive_ = true;
  std::string padding_algorithm_;

 public:
  AdaptivePoolTester(const Place& place,
                     const std::string& alias,
                     DDim dims,
                     std::string pooling_type,
                     bool global_pooling,
                     std::vector<int> strides = {1, 1},
                     std::vector<int> paddings = {0, 0},
                     std::vector<int> ksize = {2, 2},
                     bool exclusive = true,
                     bool ceil_mode = false,
                     bool adaptive = true,
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
    std::vector<int64_t> out_shape{dims_[0], dims_[1], ksize_[0], ksize_[1]};
    auto out = scope->NewTensor(out_);
    CHECK(out);
    LOG(INFO) << "outshape size is: " << out_shape[2] << ", " << out_shape[3];
    LOG(INFO) << "adaptive is: " << adaptive_;
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
    int size_out_n = in_c * ksize_[0] * ksize_[1];
    int size_out_c = ksize_[0] * ksize_[1];

    for (int n = 0; n < in_n; ++n) {
      for (int c = 0; c < in_c; ++c) {
        // Slide Window
        for (int kh = 0; kh < ksize_[0]; ++kh) {
          for (int kw = 0; kw < ksize_[1]; ++kw) {
            int hstart = std::floor(static_cast<float>(kh) * in_h / ksize_[0]);
            int hend = std::ceil(static_cast<float>(kh + 1) * in_h / ksize_[0]);
            int wstart = std::floor(static_cast<float>(kw) * in_w / ksize_[1]);
            int wend = std::ceil(static_cast<float>(kw + 1) * in_w / ksize_[1]);
            int win_size = (hend - hstart) * (wend - wstart);
            float res = 0;
            if (pooling_type_ == "max") {
              res = src_ptr[n * size_in_n + c * size_in_c + hstart * in_w +
                            wstart];
            }
            for (int i = hstart; i < hend; ++i) {
              for (int j = wstart; j < wend; ++j) {
                int src_idx = n * size_in_n + c * size_in_c + i * in_w + j;
                if (pooling_type_ == "avg") {
                  res += src_ptr[src_idx];
                } else {
                  res = std::max(res, src_ptr[src_idx]);
                }
              }
            }
            if (pooling_type_ == "avg") {
              res /= win_size;
            }
            dst_ptr[n * size_out_n + c * size_out_c + kh * ksize_[1] + kw] =
                res;
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
    // fill_data_const(din.data(), 1.f, dims_.production());
    SetCommonTensor(x_, dims_, din.data());
  }
};

void TestAdaptiveAveragePool2D(Place place, float abs_error = 2e-5) {
  for (auto dims : std::vector<std::vector<int64_t>>{{2, 3, 4, 5}}) {
    std::unique_ptr<arena::TestCase> tester(
        new AdaptivePoolTester(place, "def", DDim(dims), "avg", false));
    arena::Arena arena(std::move(tester), place, abs_error);
    arena.TestPrecision();
  }
}

void TestAdaptiveMaxPool2D(Place place, float abs_error = 2e-5) {
  for (auto dims : std::vector<std::vector<int64_t>>{{2, 3, 4, 5}}) {
    std::unique_ptr<arena::TestCase> tester(
        new AdaptivePoolTester(place, "def", DDim(dims), "max", false));
    arena::Arena arena(std::move(tester), place, abs_error);
    arena.TestPrecision();
  }
}

TEST(AdaptiveAveragePool2D, precision) {
  LOG(INFO) << "test adaptive average pool op";
  float abs_error = 2e-5;
  Place place;
#if defined(LITE_WITH_NNADAPTER)
  place = TARGET(kNNAdapter);
#if defined(NNADAPTER_WITH_HUAWEI_ASCEND_NPU)
  abs_error = 1e-2;
#elif defined(NNADAPTER_WITH_CAMBRICON_MLU)
  abs_error = 1e-3;
#elif defined(NNADAPTER_WITH_INTEL_OPENVINO)
  abs_error = 1e-3;
#else
  return;
#endif
#else
  return;
#endif

  TestAdaptiveAveragePool2D(place, abs_error);
}

TEST(AdaptiveMaxPool2D, precision) {
  LOG(INFO) << "test adaptive average pool op";
  float abs_error = 2e-5;
  Place place;
#if defined(LITE_WITH_NNADAPTER)
  place = TARGET(kNNAdapter);
#if defined(NNADAPTER_WITH_HUAWEI_ASCEND_NPU)
  abs_error = 1e-2;
  return;
#else
  return;
#endif
#else
  return;
#endif

  TestAdaptiveMaxPool2D(place, abs_error);
}

}  // namespace lite
}  // namespace paddle
