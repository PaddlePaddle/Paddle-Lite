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
#include <cstring>
#include "lite/api/paddle_use_kernels.h"
#include "lite/api/paddle_use_ops.h"
#include "lite/core/test/arena/framework.h"
#include "lite/tests/utils/fill_data.h"

namespace paddle {
namespace lite {

class FillAnyLikeComputeTester : public arena::TestCase {
 protected:
  std::string x_ = "x";
  std::string out_ = "out";
  DDim x_dims_;
  float value_;
  int dtype_;

 public:
  FillAnyLikeComputeTester(const Place& place,
                           const std::string& alias,
                           const DDim& x_dims,
                           const float value = 1.f,
                           const int dtype = 5)
      : TestCase(place, alias), x_dims_(x_dims), value_(value), dtype_(dtype) {}

  void RunBaseline(Scope* scope) override {
    auto* out = scope->NewTensor(out_);
    auto* x = scope->FindTensor(x_);
    out->Resize(x_dims_);
    out->set_lod(x->lod());

    switch (dtype_) {
      case 2: {
        auto value = static_cast<int>(value_);
        auto* out_data = out->template mutable_data<int>();
        for (int64_t i = 0; i < out->numel(); i++) {
          out_data[i] = value;
        }
        break;
      }
      case 3: {
        auto value = static_cast<int64_t>(value_);
        auto* out_data = out->template mutable_data<int64_t>();
        for (int64_t i = 0; i < out->numel(); i++) {
          out_data[i] = value;
        }
        break;
      }
      case -1:  // same as input dtype
      case 5: {
        auto value = static_cast<float>(value_);
        auto* out_data = out->template mutable_data<float>();
        for (int64_t i = 0; i < out->numel(); i++) {
          out_data[i] = value;
        }
        break;
      }
      default:
        LOG(ERROR) << "Unsupported dtype: " << dtype_;
        break;
    }
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType("fill_any_like");
    op_desc->SetInput("X", {x_});
    op_desc->SetAttr("value", value_);
    op_desc->SetAttr("dtype", dtype_);
    op_desc->SetOutput("Out", {out_});
  }

  void PrepareData() override {
    std::vector<float> x(x_dims_.production());
    fill_data_rand(x.data(), -1.f, 1.f, x_dims_.production());
    SetCommonTensor(x_, x_dims_, x.data());
  }
};

void TestFillAnyLike(Place place,
                     float abs_error,
                     float value = 1.f,
                     int dtype = 5) {
  std::vector<std::vector<int64_t>> x_shapes{
      {2, 3, 4, 5}, {2, 3, 4}, {3, 4}, {4}};
  for (auto x_shape : x_shapes) {
    std::unique_ptr<arena::TestCase> tester(new FillAnyLikeComputeTester(
        place, "def", DDim(x_shape), value, dtype));
    arena::Arena arena(std::move(tester), place, abs_error);
    arena.TestPrecision();
  }
}

TEST(fill_any_like, precision) {
  Place place;
  float abs_error = 1e-5;
#if defined(LITE_WITH_NNADAPTER)
  place = TARGET(kNNAdapter);
#if defined(NNADAPTER_WITH_HUAWEI_ASCEND_NPU)
  abs_error = 5e-2;
#elif defined(NNADAPTER_WITH_VERISILICON_TIMVX)
  abs_error = 5e-2;
  TestFillAnyLike(place, abs_error, 1.f, -1);
  return;
#elif defined(NNADAPTER_WITH_CAMBRICON_MLU)
  abs_error = 1e-5;
#elif defined(NNADAPTER_WITH_INTEL_OPENVINO)
  abs_error = 1e-5;
#else
  return;
#endif
#else
  return;
#endif

  TestFillAnyLike(place, abs_error, 1.f, -1);
  TestFillAnyLike(place, abs_error, 1.f, 2);
  // TestFillAnyLike(place, abs_error, 1.f, 3);
  TestFillAnyLike(place, abs_error, 1.f, 5);
}

}  // namespace lite
}  // namespace paddle
