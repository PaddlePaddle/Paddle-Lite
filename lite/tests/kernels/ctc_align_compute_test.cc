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

class CtcAlignComputeTester : public arena::TestCase {
 protected:
  // common attributes for this op.
  std::string input_ = "input";
  std::string input_length_ = "input_length";
  std::string output_ = "output";
  std::string output_length_ = "output_length";
  std::vector<int> input_data_;
  std::vector<int64_t> input_shape_;
  std::vector<std::vector<uint64_t>> input_lod_;
  std::vector<int> input_length_data_;
  std::vector<int64_t> input_length_shape_;
  std::vector<int> output_data_;
  std::vector<int64_t> output_shape_;
  std::vector<std::vector<uint64_t>> output_lod_;
  std::vector<int> output_length_data_;
  std::vector<int64_t> output_length_shape_;
  int blank_;
  bool merge_repeated_;
  int padding_value_;

 public:
  CtcAlignComputeTester(const Place& place,
                        const std::string& alias,
                        const std::vector<int>& input_data,
                        const std::vector<int64_t> input_shape,
                        const std::vector<std::vector<uint64_t>>& input_lod,
                        const std::vector<int>& input_length_data,
                        const std::vector<int64_t> input_length_shape,
                        const int blank,
                        const bool merge_repeated,
                        const int padding_value,
                        const std::vector<int>& output_data,
                        const std::vector<int64_t>& output_shape,
                        const std::vector<std::vector<uint64_t>>& output_lod,
                        const std::vector<int>& output_length_data,
                        const std::vector<int64_t>& output_length_shape)
      : TestCase(place, alias) {
    input_data_ = input_data;
    input_shape_ = input_shape;
    input_lod_ = input_lod;
    input_length_data_ = input_length_data;
    input_length_shape_ = input_length_shape;
    blank_ = blank;
    merge_repeated_ = merge_repeated;
    padding_value_ = padding_value;
    output_data_ = output_data;
    output_shape_ = output_shape;
    output_lod_ = output_lod;
    output_length_data_ = output_length_data;
    output_length_shape_ = output_length_shape;
  }

  void RunBaseline(Scope* scope) override {
    auto* output_tensor = scope->NewTensor(output_);
    output_tensor->Resize(output_shape_);
    if (!output_lod_.empty()) {
      output_tensor->set_lod(output_lod_);
    }
    auto* output_data = output_tensor->mutable_data<int>();
    int64_t output_num = 1;
    for (auto e : output_shape_) {
      output_num *= e;
    }
    for (int i = 0; i < output_num; i++) {
      output_data[i] = output_data_[i];
    }

    if (!input_length_data_.empty() && !output_length_data_.empty()) {
      auto* output_length_tensor = scope->NewTensor(output_length_);
      output_length_tensor->Resize(output_length_shape_);
      auto* output_length_data = output_length_tensor->mutable_data<int>();
      int64_t num = 1;
      for (auto e : output_length_shape_) {
        num *= e;
      }
      for (int i = 0; i < num; i++) {
        output_length_data[i] = output_length_data_[i];
      }
    }
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType("ctc_align");
    op_desc->SetInput("Input", {input_});
    op_desc->SetOutput("Output", {output_});
    if (!input_length_data_.empty()) {
      op_desc->SetInput("InputLength", {input_length_});
      op_desc->SetOutput("OutputLength", {output_length_});
    }
    op_desc->SetAttr("blank", blank_);
    op_desc->SetAttr("merge_repeated", merge_repeated_);
    op_desc->SetAttr("padding_value", padding_value_);
  }

  void PrepareData() override {
    SetCommonTensor(input_, DDim(input_shape_), input_data_.data(), input_lod_);
    if (!input_length_data_.empty()) {
      SetCommonTensor(
          input_length_, DDim(input_length_shape_), input_length_data_.data());
    }
  }
};
TEST(CtcAlign1, precision) {
  LOG(INFO) << "test ctc_align op";
#ifdef LITE_WITH_ARM
  // Define variable
  const std::vector<int>& input_data = {
      0, 1, 2, 2, 0, 4, 0, 4, 5, 0, 6, 6, 0, 0, 7, 7, 7, 0};
  const std::vector<int64_t> input_shape = {18, 1};
  const std::vector<std::vector<uint64_t>> input_lod = {{0, 11, 18}};
  const std::vector<int> input_length_data = {};
  const std::vector<int64_t> input_length_shape = {};
  const int blank = 0;
  const bool merge_repeated = false;
  const int padding_value = 0;
  const std::vector<int> output_data = {1, 2, 2, 4, 4, 5, 6, 6, 7, 7, 7};
  const std::vector<int64_t> output_shape = {11, 1};
  const std::vector<std::vector<uint64_t>> output_lod = {{0, 7, 11}};
  const std::vector<int> output_length_data = {};
  const std::vector<int64_t> output_length_shape = {};

  // Test
  Place place(TARGET(kHost), PRECISION(kInt32));
  std::unique_ptr<arena::TestCase> tester(
      new CtcAlignComputeTester(place,
                                "def",
                                input_data,
                                input_shape,
                                input_lod,
                                input_length_data,
                                input_length_shape,
                                blank,
                                merge_repeated,
                                padding_value,
                                output_data,
                                output_shape,
                                output_lod,
                                output_length_data,
                                output_length_shape));
  arena::Arena arena(std::move(tester), place, 2e-5);
  arena.TestPrecision();
#endif
}

TEST(CtcAlign2, precision) {
  LOG(INFO) << "test ctc_align op";
#ifdef LITE_WITH_ARM
  // Define variable
  const std::vector<int>& input_data = {
      0, 1, 2, 2, 0, 4, 0, 4, 5, 0, 6, 0, 0, 7, 7, 7, 0, 0};
  const std::vector<int64_t> input_shape = {3, 6};
  const std::vector<std::vector<uint64_t>> input_lod = {};
  const std::vector<int> input_length_data = {6, 5, 4};
  const std::vector<int64_t> input_length_shape = {3, 1};
  const int blank = 0;
  const bool merge_repeated = true;
  const int padding_value = 0;
  const std::vector<int> output_data = {
      1, 2, 4, 0, 0, 0, 4, 5, 6, 0, 0, 0, 7, 0, 0, 0, 0, 0};
  const std::vector<int64_t> output_shape = {3, 6};
  const std::vector<std::vector<uint64_t>> output_lod = {};
  const std::vector<int> output_length_data = {3, 3, 1};
  const std::vector<int64_t> output_length_shape = {3, 1};

  // Test
  Place place(TARGET(kHost), PRECISION(kInt32));
  std::unique_ptr<arena::TestCase> tester(
      new CtcAlignComputeTester(place,
                                "def",
                                input_data,
                                input_shape,
                                input_lod,
                                input_length_data,
                                input_length_shape,
                                blank,
                                merge_repeated,
                                padding_value,
                                output_data,
                                output_shape,
                                output_lod,
                                output_length_data,
                                output_length_shape));
  arena::Arena arena(std::move(tester), place, 2e-5);
  arena.TestPrecision();
#endif
}

TEST(CtcAlign3, precision) {
  LOG(INFO) << "test ctc_align op";
#ifdef LITE_WITH_ARM
  // Define variable
  const std::vector<int>& input_data = {
      0, 1, 2, 2, 0, 4, 0, 4, 5, 0, 6, 0, 0, 7, 7, 7, 0, 0};
  const std::vector<int64_t> input_shape = {3, 6};
  const std::vector<std::vector<uint64_t>> input_lod = {};
  const std::vector<int> input_length_data = {6, 5, 4};
  const std::vector<int64_t> input_length_shape = {3, 1};
  const int blank = 0;
  const bool merge_repeated = false;
  const int padding_value = 0;
  const std::vector<int> output_data = {
      1, 2, 2, 4, 0, 0, 4, 5, 6, 0, 0, 0, 7, 7, 7, 0, 0, 0};
  const std::vector<int64_t> output_shape = {3, 6};
  const std::vector<std::vector<uint64_t>> output_lod = {};
  const std::vector<int> output_length_data = {4, 3, 3};
  const std::vector<int64_t> output_length_shape = {3, 1};

  // Test
  Place place(TARGET(kHost), PRECISION(kInt32));
  std::unique_ptr<arena::TestCase> tester(
      new CtcAlignComputeTester(place,
                                "def",
                                input_data,
                                input_shape,
                                input_lod,
                                input_length_data,
                                input_length_shape,
                                blank,
                                merge_repeated,
                                padding_value,
                                output_data,
                                output_shape,
                                output_lod,
                                output_length_data,
                                output_length_shape));
  arena::Arena arena(std::move(tester), place, 2e-5);
  arena.TestPrecision();
#endif
}
}  // namespace lite
}  // namespace paddle
