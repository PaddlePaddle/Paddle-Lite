// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

template <class T = float>
static void basefunc_NCHW(const T* input,
                          T* output,
                          const int ntchw,
                          const int tchw,
                          const int chw,
                          const int hw,
                          const int t,
                          const int c1,
                          const int c2) {
  int src_it = 0;
  for (int i = 0; i < ntchw; i++) {
    int it = (i % tchw) / chw;
    int ic = (i % chw) / hw;
    if (ic < c1) {
      src_it = it - 1;
    } else if (ic < c2) {
      src_it = it + 1;
    } else {
      src_it = it;
    }
    if (src_it < 0 || src_it >= t) {
      output[i] = 0;
    } else {
      output[i] = input[i + (src_it - it) * chw];
    }
  }
}

template <class T = float>
static void basefunc_NHWC(const T* input,
                          T* output,
                          const int nthwc,
                          const int thwc,
                          const int hwc,
                          const int t,
                          const int c,
                          const int c1,
                          const int c2) {
  int src_it = 0;
  for (int i = 0; i < nthwc; i++) {
    int it = (i % thwc) / hwc;
    int ic = i % c;

    if (ic < c1) {
      src_it = it - 1;
    } else if (ic < c2) {
      src_it = it + 1;
    } else {
      src_it = it;
    }

    if (src_it < 0 || src_it >= t) {
      output[i] = 0;
    } else {
      output[i] = input[i + (src_it - it) * hwc];
    }
  }
}

template <class T = float>
class TemporalShiftComputeTester : public arena::TestCase {
 protected:
  // common attributes for this op.
  std::string x_ = "X";
  std::string out_ = "Out";
  DDim dims_;
  int seg_num_ = 2;
  float shift_ratio_ = 0.25f;
  std::string data_format_{"NCHW"};

 public:
  TemporalShiftComputeTester(const Place& place,
                             const std::string& alias,
                             int seg_num,
                             float shift_ratio,
                             const DDim& dims,
                             std::string data_format)
      : TestCase(place, alias),
        seg_num_(seg_num),
        shift_ratio_(shift_ratio),
        dims_(dims),
        data_format_(data_format) {}

  void RunBaseline(Scope* scope) override {
    auto* out = scope->NewTensor(out_);
    CHECK(out);
    out->Resize(dims_);

    auto out_data = out->template mutable_data<T>();
    auto* x = scope->FindTensor(x_);
    auto x_data = x->template data<T>();

    const int nt = dims_[0];
    const int t = seg_num_;
    const int c = data_format_ == "NCHW" ? dims_[1] : dims_[3];
    const int h = data_format_ == "NCHW" ? dims_[2] : dims_[1];
    const int w = data_format_ == "NCHW" ? dims_[3] : dims_[2];

    const int hw = h * w;
    const int chw = c * hw;
    const int tchw = t * chw;
    const int ntchw = nt * chw;

    const int c1 = c * shift_ratio_;
    const int c2 = c * 2 * shift_ratio_;

    if (data_format_ == "NCHW") {
      basefunc_NCHW(x_data, out_data, ntchw, tchw, chw, hw, t, c1, c2);
    } else {
      basefunc_NHWC(x_data, out_data, ntchw, tchw, chw, t, c, c1, c2);
    }
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) override {
    op_desc->SetType("temporal_shift");
    op_desc->SetInput("X", {x_});
    op_desc->SetOutput("Out", {out_});
    op_desc->SetAttr("seg_num", seg_num_);
    op_desc->SetAttr("shift_ratio", shift_ratio_);
    op_desc->SetAttr("data_format", data_format_);
  }

  void PrepareData() override {
    std::vector<T> x(dims_.production());
    fill_data_rand(x.data(), -1.f, 1.f, dims_.production());
    SetCommonTensor(x_, dims_, x.data());
  }
};

template <class T = float>
void TestTemporalShift(const Place& place,
                       const std::string& alias,
                       int seg_num,
                       float shift_ratio,
                       const DDim& dims,
                       std::string data_format,
                       float abs_error = 2e-5) {
  std::unique_ptr<arena::TestCase> tester(new TemporalShiftComputeTester<T>(
      place, alias, seg_num, shift_ratio, dims, data_format));
  arena::Arena arena(std::move(tester), place, abs_error);
  arena.TestPrecision();
}

template <class T = float>
void TestPer(const Place& place,
             const std::string& alias,
             int seg_num,
             float shift_ratio,
             const DDim& dims,
             std::string data_format,
             float abs_error = 2e-5) {
  std::unique_ptr<arena::TestCase> tester(new TemporalShiftComputeTester<T>(
      place, alias, seg_num, shift_ratio, dims, data_format));
  arena::Arena arena(std::move(tester), place, abs_error);
  arena.TestPerformance();
}

TEST(TS_FP32_NCHW, precision) {
  Place place;
  float abs_error = 2e-5;
#if defined(LITE_WITH_ARM) || defined(LITE_WITH_X86)
  place = TARGET(kHost);
#else
  return;
#endif
  std::string data_format = "NCHW";
  for (auto dims : std::vector<std::vector<int64_t>>{{1, 8, 3, 4}}) {
    for (float shift_ratio : {0.25f}) {
      for (int seg_num : {1, 2, 4}) {
        TestTemporalShift(place,
                          "fp32",
                          seg_num,
                          shift_ratio,
                          DDim(dims),
                          data_format,
                          abs_error);
      }
    }
  }
}

TEST(TS_FP32_NCHW, performance) {
  Place place;
  float abs_error = 2e-5;
#if defined(LITE_WITH_ARM) || defined(LITE_WITH_X86)
  place = TARGET(kHost);
#else
  return;
#endif
  std::string data_format = "NCHW";
  for (auto dims : std::vector<std::vector<int64_t>>{{1, 8, 3, 4}}) {
    for (float shift_ratio : {0.25f}) {
      for (int seg_num : {1, 2, 4}) {
        TestPer(place,
                "fp32",
                seg_num,
                shift_ratio,
                DDim(dims),
                data_format,
                abs_error);
      }
    }
  }
}

}  // namespace lite
}  // namespace paddle
