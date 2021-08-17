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
#include <cstring>
#include "lite/api/paddle_use_kernels.h"
#include "lite/api/paddle_use_ops.h"
#include "lite/core/test/arena/framework.h"
#include "lite/tests/utils/fill_data.h"

namespace paddle {
namespace lite {

template <class T>
class SequencePadTester : public arena::TestCase {
 protected:
  std::string x_ = "x";
  std::string pad_value_ = "pad_value";
  std::string out_ = "out";
  std::string length_ = "length";
  DDim x_dims_{{9, 2, 3, 4}};
  LoD x_lod_{{{0, 2, 5, 9}}};
  T value_ = 0;
  int padded_length_ = 4;

 public:
  SequencePadTester(const Place& place, const std::string& alias)
      : TestCase(place, alias) {}

  void RunBaseline(Scope* scope) override {
    auto* out = scope->NewTensor(out_);
    auto out_shape = x_dims_.Vectorize();
    out_shape[0] = padded_length_;
    out_shape.insert(out_shape.begin(),
                     static_cast<int64_t>(x_lod_[0].size() - 1));
    out->Resize(out_shape);
    auto* out_data = out->template mutable_data<T>();
    for (int64_t i = 0; i < out->numel(); i++) {
      out_data[i] = value_;
    }

    int n = x_dims_.production() / x_dims_[0];
    int out_step = padded_length_ * n;
    auto* x = scope->FindTensor(x_);
    auto* x_data = x->template data<T>();
    for (size_t i = 1; i < x_lod_[0].size(); i++) {
      int x_step = (x_lod_[0][i] - x_lod_[0][i - 1]) * n;
      memcpy(out_data, x_data, sizeof(T) * x_step);
      x_data += x_step;
      out_data += out_step;
    }

    auto* length = scope->NewTensor(length_);
    length->Resize({static_cast<int64_t>(x_lod_[0].size() - 1)});
    int64_t* length_data = length->template mutable_data<int64_t>();
    for (size_t i = 1; i < x_lod_[0].size(); i++) {
      length_data[i - 1] = x_lod_[0][i] - x_lod_[0][i - 1];
    }
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType("sequence_pad");
    op_desc->SetInput("X", {x_});
    op_desc->SetInput("PadValue", {pad_value_});
    op_desc->SetOutput("Out", {out_});
    op_desc->SetOutput("Length", {length_});
    op_desc->SetAttr("padded_length", padded_length_);
  }

  void PrepareData() override {
    std::vector<T> x_data(x_dims_.production());
    fill_data_rand<T>(x_data.data(), -10, 10, x_dims_.production());
    SetCommonTensor(x_, x_dims_, x_data.data(), x_lod_);

    std::vector<T> pad_value_data{0};
    SetCommonTensor(pad_value_, DDim{{1}}, pad_value_data.data());
  }
};

template <class T>
void TestSequencePad(const Place place,
                     const float abs_error,
                     const std::string alias) {
  std::unique_ptr<arena::TestCase> tester(
      new SequencePadTester<T>(place, alias));
  arena::Arena arena(std::move(tester), place, abs_error);
  arena.TestPrecision();
}

TEST(sequence_pad, precision) {
  Place place;
  float abs_error = 1e-5;
#if defined(LITE_WITH_ARM) || defined(LITE_WITH_X86)
  place = TARGET(kHost);
#else
  return;
#endif

  TestSequencePad<float>(place, abs_error, "def");
  TestSequencePad<int>(place, abs_error, "int32");
  TestSequencePad<int64_t>(place, abs_error, "int64");
}

}  // namespace lite
}  // namespace paddle
