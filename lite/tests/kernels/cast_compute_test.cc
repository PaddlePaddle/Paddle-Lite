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

class CastComputeTester : public arena::TestCase {
 protected:
  std::string x_ = "x";
  std::string out_ = "out";
  // BOOL = 0;INT16 = 1;INT32 = 2;INT64 = 3;FP16 = 4;FP32 = 5;FP64 = 6;
  // SIZE_T = 19;UINT8 = 20;INT8 = 21;
  int in_dtype_;
  int out_dtype_;
  DDim dims_{{2, 2}};

 public:
  CastComputeTester(const Place& place,
                    const std::string& alias,
                    int in_dtype,
                    int out_dtype)
      : TestCase(place, alias), in_dtype_(in_dtype), out_dtype_(out_dtype) {}

  template <typename T1, typename T2>
  void RunBaselineHelper(Scope* scope) {
    auto* x = scope->FindTensor(x_);
    auto* x_data = x->data<T1>();
    auto* out = scope->NewTensor(out_);
    CHECK(out);
    out->Resize(dims_);
    auto* out_data = out->mutable_data<T2>();
    for (int i = 0; i < dims_.production(); i++) {
      *out_data = static_cast<T2>(*x_data);
      out_data++;
      x_data++;
    }
  }

  void RunBaseline(Scope* scope) override {
    if (in_dtype_ == 20 && out_dtype_ == 5) {
      RunBaselineHelper<uint8_t, float>(scope);
    } else if (in_dtype_ == 2 && out_dtype_ == 5) {
      RunBaselineHelper<int32_t, float>(scope);
    } else if (in_dtype_ == 3 && out_dtype_ == 5) {
      RunBaselineHelper<int64_t, float>(scope);
    } else if (in_dtype_ == 5 && out_dtype_ == 3) {
      RunBaselineHelper<float, int64_t>(scope);
    } else if (in_dtype_ == 21 && out_dtype_ == 5) {
      RunBaselineHelper<int8_t, float>(scope);
    } else if (in_dtype_ == 5 && out_dtype_ == 21) {
      RunBaselineHelper<float, int8_t>(scope);
    } else {
      LOG(FATAL) << "unsupported";
    }
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType("cast");
    op_desc->SetInput("X", {x_});
    op_desc->SetOutput("Out", {out_});
    op_desc->SetAttr("in_dtype", in_dtype_);
    op_desc->SetAttr("out_dtype", out_dtype_);
  }

  template <typename T1>
  void PrepareDataHelper() {
    std::vector<T1> x_data(dims_.production());
    for (int i = 0; i < dims_.production(); i++) {
      x_data[i] = static_cast<T1>(i % 128);
    }
    SetCommonTensor(x_, dims_, x_data.data());
  }

  void PrepareData() override {
    // BOOL = 0;INT16 = 1;INT32 = 2;INT64 = 3;FP16 = 4;FP32 = 5;FP64 = 6;
    // SIZE_T = 19;UINT8 = 20;INT8 = 21;
    switch (in_dtype_) {
      case 20:
        PrepareDataHelper<uint8_t>();
        break;
      case 21:
        PrepareDataHelper<int8_t>();
        break;
      case 1:
        PrepareDataHelper<int16_t>();
        break;
      case 2:
        PrepareDataHelper<int32_t>();
        break;
      case 3:
        PrepareDataHelper<int64_t>();
        break;
      case 5:
        PrepareDataHelper<float>();
        break;
      case 6:
        PrepareDataHelper<double>();
        break;
      case 19:
        PrepareDataHelper<size_t>();
        break;
      default:
        LOG(FATAL) << "unsupported data type: " << in_dtype_;
        break;
    }
  }
};

void TestCast(Place place, float abs_error, int in_dtype, int out_dtype) {
  std::unique_ptr<arena::TestCase> tester(
      new CastComputeTester(place, "def", in_dtype, out_dtype));
  arena::Arena arena(std::move(tester), place, abs_error);
  arena.TestPrecision();
}

TEST(Cast, precision) {
  LOG(INFO) << "test cast op";
  Place place;
  float abs_error = 2e-5;
#if defined(LITE_WITH_ARM)
  place = TARGET(kARM);
#elif defined(LITE_WITH_XPU) && defined(LITE_WITH_XTCL)
  place = TARGET(kXPU);
#elif defined(LITE_WITH_HUAWEI_ASCEND_NPU)
  place = TARGET(kHuaweiAscendNPU);
  abs_error = 1e-2;  // precision_mode default is force_fp16
#else
  return;
#endif

// BOOL = 0;INT16 = 1;INT32 = 2;INT64 = 3;FP16 = 4;FP32 = 5;FP64 = 6;
// SIZE_T = 19;UINT8 = 20;INT8 = 21;
#if !defined(LITE_WITH_XPU) && !defined(LITE_WITH_HUAWEI_ASCEND_NPU)
  TestCast(place, abs_error, 20, 5);
#endif
  TestCast(place, abs_error, 2, 5);
#if defined(LITE_WITH_XPU) || defined(LITE_WITH_HUAWEI_ASCEND_NPU)
  TestCast(place, abs_error, 3, 5);
  TestCast(place, abs_error, 5, 3);
#endif
}

}  // namespace lite
}  // namespace paddle
