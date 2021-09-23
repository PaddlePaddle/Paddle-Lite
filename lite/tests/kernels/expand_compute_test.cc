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

template <class T,
          bool has_expandtimes = false,
          bool has_expand_times_tensor = false>
class ExpandComputeTester : public arena::TestCase {
 protected:
  // common attributes for this op.
  std::string x_ = "X";
  std::string out_ = "Out";
  std::string expandtimes_ = "ExpandTimes";
  std::vector<int> expand_times_;
  DDim dims_;

 public:
  ExpandComputeTester(const Place& place,
                      const std::string& alias,
                      const std::vector<int>& expand_times,
                      DDim dims)
      : TestCase(place, alias), expand_times_(expand_times), dims_(dims) {}

  void RunBaseline(Scope* scope) override {
    const auto* input = scope->FindTensor(x_);
    CHECK(input);
    auto* out = scope->NewTensor(out_);
    CHECK(out);

    DDim out_shape(input->dims());
    DDim in_shape = input->dims();

    for (size_t i = 0; i < expand_times_.size(); ++i) {
      out_shape[i] *= expand_times_[i];
    }
    out->Resize(out_shape);
    T* out_data = out->template mutable_data<T>();
    const T* input_data = input->template data<T>();
    std::vector<int> in_stride(in_shape.size(), 1),
        out_stride(out_shape.size(), 1);
    for (int i = in_shape.size() - 2; i >= 0; --i) {
      in_stride[i] = in_shape[i + 1] * in_stride[i + 1];
    }
    for (int i = out_shape.size() - 2; i >= 0; --i) {
      out_stride[i] = out_shape[i + 1] * out_stride[i + 1];
    }
    for (int64_t out_id = 0; out_id < out_shape.production(); ++out_id) {
      int in_id = 0;
      for (int i = expand_times_.size() - 1; i >= 0; --i) {
        int in_j = (out_id / out_stride[i]) % in_shape[i];
        in_id += in_j * in_stride[i];
      }
      out_data[out_id] = input_data[in_id];
    }
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType("expand");
    op_desc->SetInput("X", {x_});
    if (has_expandtimes) {
      op_desc->SetInput("ExpandTimes", {expandtimes_});
    }
    if (has_expand_times_tensor) {
      std::vector<std::string> expand_times_tensor_;
      for (size_t i = 0; i < expand_times_.size(); i++) {
        expand_times_tensor_.push_back("expand_times_tensor_" +
                                       paddle::lite::to_string(i));
        op_desc->SetInput("expand_times_tensor", expand_times_tensor_);
      }
    }
    op_desc->SetOutput("Out", {out_});
    op_desc->SetAttr("expand_times", expand_times_);
  }

  void PrepareData() override {
    std::vector<T> in_data(dims_.production());
    for (int i = 0; i < dims_.production(); ++i) {
      in_data[i] = i;
    }
    SetCommonTensor(x_, dims_, in_data.data());
    if (has_expandtimes) {
      SetCommonTensor(expandtimes_,
                      DDim{{static_cast<int64_t>(expand_times_.size())}},
                      expand_times_.data());
    }
    if (has_expand_times_tensor) {
      for (size_t i = 0; i < expand_times_.size(); ++i) {
        SetCommonTensor("expand_times_tensor_" + paddle::lite::to_string(i),
                        DDim({1}),
                        &expand_times_[i]);
      }
    }
  }
};

template <class T,
          bool has_expandtimes = false,
          bool has_expand_times_tensor = false>
void test_expand_3dim(Place place, float abs_error) {
  std::string alias{"def"};

  for (std::vector<int> expand_times : {std::vector<int>({2, 3, 1}),
                                        std::vector<int>({2, 2, 2}),
                                        std::vector<int>({3, 1, 2})}) {
    for (int C : {3}) {
      for (int H : {2}) {
        for (int W : {4}) {
          std::unique_ptr<arena::TestCase> tester(
              new ExpandComputeTester<T,
                                      has_expandtimes,
                                      has_expand_times_tensor>(
                  place, alias, expand_times, DDim({C, H, W})));
          arena::Arena arena(std::move(tester), place, abs_error);
          arena.TestPrecision();
        }
      }
    }
  }
}

template <class T,
          bool has_expandtimes = false,
          bool has_expand_times_tensor = false>
void test_expand_4dim(Place place, float abs_error) {
  std::string alias{"def"};

  for (std::vector<int> expand_times : {std::vector<int>({2, 3, 1, 4}),
                                        std::vector<int>({2, 2, 2, 2}),
                                        std::vector<int>({3, 1, 2, 1})}) {
    for (int N : {2}) {
      for (int C : {3}) {
        for (int H : {2}) {
          for (int W : {4}) {
            std::unique_ptr<arena::TestCase> tester(
                new ExpandComputeTester<T,
                                        has_expandtimes,
                                        has_expand_times_tensor>(
                    place, alias, expand_times, DDim({N, C, H, W})));
            arena::Arena arena(std::move(tester), place, abs_error);
            arena.TestPrecision();
          }
        }
      }
    }
  }
}

TEST(Expand, precision) {
  Place place;
  float abs_error = 1e-5;
#if defined(LITE_WITH_NPU)
  place = TARGET(kNPU);
  abs_error = 1e-2;  // Using fp16 in NPU
#elif defined(LITE_WITH_ARM) || defined(LITE_WITH_X86)
  place = Place(TARGET(kHost), PRECISION(kAny));
#else
  return;
#endif

  test_expand_3dim<float>(place, abs_error);
  test_expand_4dim<float>(place, abs_error);
#ifndef LITE_WITH_NPU
  test_expand_3dim<int>(place, abs_error);
  test_expand_4dim<int>(place, abs_error);
  test_expand_4dim<float, true>(place, abs_error);
  test_expand_4dim<float, false, true>(place, abs_error);
  test_expand_4dim<int, true, true>(place, abs_error);
#endif
}

}  // namespace lite
}  // namespace paddle
