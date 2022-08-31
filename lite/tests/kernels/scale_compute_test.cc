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
#include "lite/core/test/arena/framework.h"
#include "lite/tests/utils/fill_data.h"

namespace paddle {
namespace lite {

template <typename T>
class ScaleComputeTester : public arena::TestCase {
 protected:
  // common attributes for this op.
  std::string x_ = "x";
  std::string out_ = "out";
  DDim x_dims_{{100, 20}};
  float scale_ = 0.;
  float bias_ = 0.;
  bool bias_after_scale_ = true;
  bool have_relu6 = false;
  std::string act_type_ = "relu6";
  float alpha_ = 6.0f;

 public:
  ScaleComputeTester(const Place& place,
                     const std::string& alias,
                     const DDim& x_dims,
                     float scale,
                     float bias,
                     bool bias_after_scale = true,
                     bool have_relu6 = false)
      : TestCase(place, alias),
        x_dims_(x_dims),
        scale_(scale),
        bias_(bias),
        bias_after_scale_(bias_after_scale),
        have_relu6(have_relu6) {}

  void RunBaseline(Scope* scope) override {
    auto* x = scope->FindTensor(x_);
    auto* x_data = x->template data<T>();
    auto* out = scope->NewTensor(out_);
    out->Resize(x_dims_);

    T scale = static_cast<T>(scale_);
    T bias = static_cast<T>(bias_);
    if (!bias_after_scale_) {
      bias *= scale;
    }

    auto out_data = out->template mutable_data<T>();
    for (int i = 0; i < x_dims_.production(); i++) {
      out_data[i] = x_data[i] * scale + bias;
    }
    if (have_relu6) {
      for (int i = 0; i < x_dims_.production(); i++) {
        out_data[i] = out_data[i] > 0.f ? out_data[i] : 0.f;
        out_data[i] = out_data[i] < 6.0f ? out_data[i] : 6.0f;
      }
    }
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType("scale");
    op_desc->SetInput("X", {x_});
    op_desc->SetOutput("Out", {out_});
    op_desc->SetAttr("scale", scale_);
    op_desc->SetAttr("bias", bias_);
    op_desc->SetAttr("bias_after_scale", bias_after_scale_);
    if (have_relu6) {
      op_desc->SetAttr("activation_type", act_type_);
      op_desc->SetAttr("alpha", alpha_);
    }
  }

  void PrepareDataHelper() {
    std::vector<T> dx(x_dims_.production());
    fill_data_rand<T>(dx.data(), -10, 10, x_dims_.production());
    SetCommonTensor(x_, x_dims_, dx.data());
  }

  void PrepareData() override {
    std::vector<T> dx(x_dims_.production());
    fill_data_rand<T>(dx.data(),
                      static_cast<T>(-10),
                      static_cast<T>(10),
                      x_dims_.production());
    SetCommonTensor(x_, x_dims_, dx.data());
  }
};

void TestScaleShape(Place place, float abs_error) {
  for (auto x_dims : std::vector<std::vector<int64_t>>{
           {5, 2, 3, 4}, {8, 3, 5}, {12, 3}, {5}}) {
    std::unique_ptr<arena::TestCase> tester(new ScaleComputeTester<float>(
        place, "def", DDim(x_dims), 1.5f, 0.2f, true));
    arena::Arena arena(std::move(tester), place, abs_error);
    arena.TestPrecision();
  }
}

void TestScaleValue(Place place, float abs_error) {
  for (float scale : {0.123, 0., -1.2}) {
    for (float bias : {1., 0., -1.2331}) {
      std::unique_ptr<arena::TestCase> tester(new ScaleComputeTester<float>(
          place, "def", DDim({5, 2, 3, 4}), scale, bias));
      arena::Arena arena(std::move(tester), place, abs_error);
      arena.TestPrecision();
    }
  }
}

void TestScaleOrder(Place place, float abs_error) {
  for (bool bias_after_scale : {true, false}) {
    std::unique_ptr<arena::TestCase> tester(new ScaleComputeTester<float>(
        place, "def", DDim({2, 3, 4, 5}), 1.5f, 0.2f, bias_after_scale));
    arena::Arena arena(std::move(tester), place, abs_error);
    arena.TestPrecision();
  }
}

template <typename T>
void TestScaleDtype(Place place, float abs_error, std::string alias) {
  std::unique_ptr<arena::TestCase> tester(new ScaleComputeTester<T>(
      place, alias, DDim({2, 3, 4, 5}), 2.f, 1.f, true, false));
  arena::Arena arena(std::move(tester), place, abs_error);
  arena.TestPrecision();
}

void TestScaleRelu6(Place place, float abs_error) {
  for (auto x_dims :
       std::vector<std::vector<int64_t>>{{5, 2, 3, 4}, {8, 3, 5}, {12, 3}}) {
    for (bool bias_after_scale : {true, false}) {
      for (bool have_relu6 : {true, false}) {
        std::unique_ptr<arena::TestCase> tester(
            new ScaleComputeTester<float>(place,
                                          "def",
                                          DDim(x_dims),
                                          1.5f,
                                          0.2f,
                                          bias_after_scale,
                                          have_relu6));
        arena::Arena arena(std::move(tester), place, abs_error);
        arena.TestPrecision();
      }
    }
  }
}

TEST(Scale, precision) {
  Place place;
  float abs_error = 2e-5;
#if defined(LITE_WITH_NNADAPTER)
  place = TARGET(kNNAdapter);
#if defined(NNADAPTER_WITH_HUAWEI_ASCEND_NPU)
  abs_error = 1e-1;
#elif defined(NNADAPTER_WITH_VERISILICON_TIMVX)
  abs_error = 1e-1;
#elif defined(NNADAPTER_WITH_NVIDIA_TENSORRT)
  abs_error = 2e-5;
#elif defined(NNADAPTER_WITH_INTEL_OPENVINO)
  abs_error = 1e-5;
#else
  return;
#endif
#elif defined(LITE_WITH_NPU)
  place = TARGET(kNPU);
  abs_error = 1e-1;  // Using fp16 in NPU
#elif defined(LITE_WITH_OPENCL)
  place = Place(TARGET(kOpenCL), PRECISION(kFP16), DATALAYOUT(kNCHW));
  abs_error = 5e-2;  // Using fp16 in OPENCL
#elif defined(LITE_WITH_ARM)
  place = TARGET(kARM);
#elif defined(LITE_WITH_X86)
  place = TARGET(kX86);
#else
  return;
#endif

  TestScaleShape(place, abs_error);
  TestScaleValue(place, abs_error);
  TestScaleOrder(place, abs_error);
#if defined(LITE_WITH_OPENCL)
  place = Place(TARGET(kOpenCL), PRECISION(kFP16), DATALAYOUT(kImageDefault));
  TestScaleShape(place, abs_error);
  TestScaleValue(place, abs_error);
  TestScaleOrder(place, abs_error);
  TestScaleRelu6(place, abs_error);
#endif
}

TEST(Scale, performance) {
  Place place;
#if defined(LITE_WITH_ARM)
  place = TARGET(kARM);
#elif defined(LITE_WITH_X86)
  place = TARGET(kX86);
#else
  return;
#endif

  std::unique_ptr<arena::TestCase> tester(new ScaleComputeTester<float>(
      place, "def", DDim(std::vector<int64_t>{5, 2, 3, 4}), 1.2, 1.1, true));

  // To modify the arm context, one can retrive the context as follows.
  // #ifdef LITE_WITH_ARM
  //   tester->context()->As<ARMContext>();
  // #endif

  arena::Arena arena(std::move(tester), place, 2e-5);
  arena.TestPerformance(100);
}

TEST(Scale, dtype) {
  Place place;
  float abs_error = 1e-4;
#if defined(LITE_WITH_ARM)
  place = TARGET(kARM);
#elif defined(LITE_WITH_X86)
  place = TARGET(kX86);
#else
  return;
#endif

  TestScaleDtype<int>(place, abs_error, "int32");
  TestScaleDtype<int64_t>(place, abs_error, "int64");
}

}  // namespace lite
}  // namespace paddle
