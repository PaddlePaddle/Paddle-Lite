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

namespace paddle {
namespace lite {

class AssignValueComputeTester : public arena::TestCase {
 protected:
  // common attributes for this op.
  std::string out_ = "out";
  int dtype_{};
  std::vector<int> shape_{};
  std::vector<int> int32_values_{};
  std::vector<float> fp32_values_{};
  size_t num_ = 1;

 public:
  AssignValueComputeTester(const Place& place,
                           const std::string& alias,
                           int dtype,
                           int n,
                           int c,
                           int h,
                           int w)
      : TestCase(place, alias) {
    dtype_ = dtype;
    shape_.push_back(n);
    shape_.push_back(c);
    shape_.push_back(h);
    shape_.push_back(w);
    num_ = n * c * h * w;
  }

  void RunBaseline(Scope* scope) override {
    auto* out = scope->NewTensor(out_);
    CHECK(out);
    std::vector<int64_t> out_shape(shape_.begin(), shape_.end());
    out->Resize(out_shape);
    if (dtype_ == 2) {
      auto* out_data = out->mutable_data<int>();
      for (int i = 0; i < out->numel(); i++) {
        out_data[i] = int32_values_[i];
      }
    } else if (dtype_ == 5) {
      auto* out_data = out->mutable_data<float>();
      for (int i = 0; i < out->numel(); i++) {
        out_data[i] = fp32_values_[i];
      }
    } else {
      LOG(FATAL) << "unsuport dtype_:" << dtype_;
    }
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType("assign_value");
    op_desc->SetAttr("shape", shape_);
    op_desc->SetAttr("dtype", dtype_);
    op_desc->SetAttr("fp32_values", fp32_values_);
    op_desc->SetAttr("int32_values", int32_values_);
    op_desc->SetOutput("Out", {out_});
  }

  void PrepareData() override {
    // int32
    if (dtype_ == 2) {
      int32_values_.resize(num_);
      for (int i = 0; i < num_; i++) {
        int32_values_[i] = i;
      }
    } else if (dtype_ == 5) {
      fp32_values_.resize(num_);
      for (int i = 0; i < num_; i++) {
        fp32_values_[i] = i / 1.23f;
      }
    } else {
      LOG(FATAL) << "unsupport dtype_:" << dtype_;
    }
  }
};

TEST(AssignValue, precision) {
  LOG(INFO) << "test argmax op";
#ifdef LITE_WITH_ARM
  LOG(INFO) << "test argmax arm";
  Place place(TARGET(kARM));

  for (int dtype : {2, 5}) {
    for (int n : {1}) {
      for (int c : {2}) {
        for (int h : {1}) {
          for (int w : {2}) {
            std::unique_ptr<arena::TestCase> tester(
                new AssignValueComputeTester(place, "def", dtype, n, c, h, w));
            arena::Arena arena(std::move(tester), place, 2e-5);
            arena.TestPrecision();
          }
        }
      }
    }
  }
#endif
}

}  // namespace lite
}  // namespace paddle
