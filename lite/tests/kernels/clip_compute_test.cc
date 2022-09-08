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

namespace paddle {
namespace lite {

class ClipComputeTester : public arena::TestCase {
 protected:
  // common attributes for this op.
  std::string x_ = "x";
  std::string out_ = "out";
  std::string min_tensor_ = "min_tensor";
  std::string max_tensor_ = "max_tensor";
  float min_{};
  float max_{};
  bool use_minmax_tensor_{};
  DDim x_dims_;

 public:
  ClipComputeTester(const Place& place,
                    const std::string& alias,
                    int n,
                    int c,
                    int h,
                    int w,
                    float min,
                    float max,
                    bool use_minmax_tensor)
      : TestCase(place, alias) {
    x_dims_ = DDim(std::vector<int64_t>({n, c, h, w}));
    min_ = min;
    max_ = max;
    use_minmax_tensor_ = use_minmax_tensor;
  }

  void RunBaseline(Scope* scope) override {
    auto* x = scope->FindTensor(x_);
    auto* out = scope->NewTensor(out_);
    CHECK(out);
    out->Resize(x->dims());
    const auto* x_data = x->data<float>();
    auto* out_data = out->mutable_data<float>();

    for (int i = 0; i < x->numel(); i++) {
      if (x_data[i] < min_)
        out_data[i] = min_;
      else if (x_data[i] > max_)
        out_data[i] = max_;
      else
        out_data[i] = x_data[i];
    }
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType("clip");
    op_desc->SetInput("X", {x_});
    op_desc->SetOutput("Out", {out_});
    if (use_minmax_tensor_) {
      op_desc->SetInput("Min", {min_tensor_});
      op_desc->SetInput("Max", {max_tensor_});
      op_desc->SetAttr("min", 0.f);
      op_desc->SetAttr("max", 0.f);
    } else {
      op_desc->SetAttr("min", min_);
      op_desc->SetAttr("max", max_);
    }
  }

  void PrepareData() override {
    std::vector<float> x_data(x_dims_.production());
    for (int i = 0; i < x_dims_.production(); i++) {
      float sign = i % 3 == 0 ? -1.0f : 1.0f;
      x_data[i] = sign * static_cast<float>(i % 128) * 0.013f + 0.001;
    }
    SetCommonTensor(x_, x_dims_, x_data.data());

    if (use_minmax_tensor_) {
      std::vector<float> min_data = {min_};
      SetCommonTensor(
          min_tensor_, DDim(std::vector<int64_t>({1})), min_data.data());

      std::vector<float> max_data = {max_};
      SetCommonTensor(
          max_tensor_, DDim(std::vector<int64_t>({1})), max_data.data());
    }
  }
};

TEST(Clip, precision) {
  LOG(INFO) << "test clip op";
  Place place;
  float abs_err = 2e-5;
  std::vector<bool> use_minmax_tensor{false, true};
#if defined(LITE_WITH_NNADAPTER)
  place = TARGET(kNNAdapter);
#if defined(NNADAPTER_WITH_HUAWEI_ASCEND_NPU)
  abs_err = 1e-2;
#elif defined(NNADAPTER_WITH_VERISILICON_TIMVX)
  abs_err = 1e-5;
  use_minmax_tensor = std::vector<bool>{false};
#elif defined(NNADAPTER_WITH_CAMBRICON_MLU)
  abs_err = 1e-5;
  use_minmax_tensor = std::vector<bool>{false};
#elif defined(NNADAPTER_WITH_HUAWEI_KIRIN_NPU)
  abs_err = 1e-3;
  use_minmax_tensor = std::vector<bool>{false};
#elif defined(NNADAPTER_WITH_NVIDIA_TENSORRT)
  abs_err = 1e-5;
  use_minmax_tensor = std::vector<bool>{false};
#elif defined(NNADAPTER_WITH_INTEL_OPENVINO)
  abs_err = 1e-5;
  use_minmax_tensor = std::vector<bool>{false};
#elif defined(NNADAPTER_WITH_QUALCOMM_QNN)
  abs_err = 1e-2;
  use_minmax_tensor = std::vector<bool>{false};
#else
  return;
#endif
#elif defined(LITE_WITH_OPENCL)
  place = Place(TARGET(kOpenCL), PRECISION(kFP16), DATALAYOUT(kImageDefault));
  abs_err = 1e-2;  // Using fp16 in OPENCL
  use_minmax_tensor = std::vector<bool>{false};
#elif defined(LITE_WITH_ARM)
  place = Place(TARGET(kARM));
#elif defined(LITE_WITH_X86)
  place = Place(TARGET(kX86));
#else
  return;
#endif

  float min = -1;
  float max = 1;
  for (int n : {1, 3}) {
    for (int c : {3, 5}) {
      for (int h : {5, 6}) {
        for (int w : {6, 7}) {
          for (bool is_use_minmax_tensor : use_minmax_tensor) {
            LOG(INFO) << "nchw:" << n << "," << c << "," << h << "," << w
                      << ",use_minmax:" << is_use_minmax_tensor;
            std::unique_ptr<arena::TestCase> tester(new ClipComputeTester(
                place, "def", n, c, h, w, min, max, is_use_minmax_tensor));
            arena::Arena arena(std::move(tester), place, abs_err);
            arena.TestPrecision();
          }
        }
      }
    }
  }
}

}  // namespace lite
}  // namespace paddle
