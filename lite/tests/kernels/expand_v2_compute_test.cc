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

template <class T>
class ExpandV2ComputeTester : public arena::TestCase {
 protected:
  // common attributes for this op.
  std::string x_ = "X";
  std::string out_ = "Out";
  std::string shape_tensor_;
  std::vector<std::string> shape_tensor_list_;
  DDim x_dims_;
  std::vector<int> expand_shape_;

 public:
  ExpandV2ComputeTester(const Place& place,
                        const std::string& alias,
                        const DDim& x_dims,
                        const std::vector<int>& expand_shape,
                        const bool use_shape_tensor = false,
                        const bool use_shape_tensor_list = false)
      : TestCase(place, alias), x_dims_(x_dims), expand_shape_(expand_shape) {
    if (use_shape_tensor) {
      shape_tensor_ = "shape";
    }
    if (use_shape_tensor_list) {
      for (size_t i = 0; i < expand_shape.size(); i++) {
        shape_tensor_list_.push_back("expand_shape_" + std::to_string(i));
      }
    }
  }

  void RunBaseline(Scope* scope) override {
    std::vector<int64_t> x_dims_shape = x_dims_.Vectorize();
    std::vector<int> x_shape(x_dims_shape.begin(), x_dims_shape.end());
    std::vector<int> expand_shape = expand_shape_;
    x_shape.insert(x_shape.begin(), expand_shape.size() - x_shape.size(), 1);
    std::vector<int> repeat_times(x_shape.size());
    for (size_t i = 0; i < x_shape.size(); ++i) {
      if (expand_shape[i] == -1) {
        expand_shape[i] = x_shape[i];
      }
      CHECK_GE(expand_shape[i], x_shape[i]);
      CHECK_EQ(expand_shape[i] % x_shape[i], 0);
      repeat_times[i] = expand_shape[i] / x_shape[i];
    }

    const auto* x = scope->FindTensor(x_);
    auto* out = scope->NewTensor(out_);
    std::vector<int64_t> out_shape(expand_shape.begin(), expand_shape.end());
    out->Resize(DDim(out_shape));
    const T* src = x->template data<T>();
    T* dst = out->template mutable_data<T>();
    DDim new_in_shape(std::vector<int64_t>(x_shape.begin(), x_shape.end()));
    int rank = repeat_times.size();
    int inner_num = 1;
    int index = rank - 1;
    int outer_num = new_in_shape.count(0, index);
    inner_num *= new_in_shape[index];
    for (int j = 0; j < outer_num; ++j) {
      for (int k = 0; k < repeat_times[index]; ++k) {
        memcpy(dst + (j * repeat_times[index] + k) * inner_num,
               src + j * inner_num,
               sizeof(T) * inner_num);
      }
    }
    inner_num *= repeat_times[index];
    for (int index = rank - 2; index >= 0; --index) {
      int outer_num = new_in_shape.count(0, index);
      inner_num *= new_in_shape[index];
      for (int j = outer_num - 1; j >= 0; --j) {
        for (int k = repeat_times[index] - 1; k >= 0; --k) {
          memcpy(dst + (j * repeat_times[index] + k) * inner_num,
                 dst + j * inner_num,
                 sizeof(T) * inner_num);
        }
      }
      inner_num *= repeat_times[index];
    }
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType("expand_v2");
    op_desc->SetInput("X", {x_});
    if (!shape_tensor_.empty()) {
      op_desc->SetInput("Shape", {shape_tensor_});
    }
    if (!shape_tensor_list_.empty()) {
      op_desc->SetInput("expand_shapes_tensor", shape_tensor_list_);
    }
    op_desc->SetOutput("Out", {out_});
    if (shape_tensor_.empty() && shape_tensor_list_.empty()) {
      op_desc->SetAttr("shape", expand_shape_);
    } else {
      op_desc->SetAttr("shape", std::vector<int>{});
    }
  }

  void PrepareData() override {
    std::vector<T> x_data(x_dims_.production());
    fill_data_rand(x_data.data(),
                   static_cast<T>(-5),
                   static_cast<T>(5),
                   x_dims_.production());
    SetCommonTensor(x_, x_dims_, x_data.data());

    if (!shape_tensor_.empty()) {
      SetCommonTensor(shape_tensor_,
                      DDim({static_cast<int64_t>(expand_shape_.size())}),
                      expand_shape_.data(),
                      {},
                      true);
    }

    if (!shape_tensor_list_.empty()) {
      for (size_t i = 0; i < expand_shape_.size(); i++) {
        SetCommonTensor(shape_tensor_list_[i],
                        DDim({1}),
                        expand_shape_.data() + i,
                        {},
                        true);
      }
    }
  }
};

template <class T>
void TestExpandV2(Place place,
                  float abs_error,
                  const std::vector<int64_t>& x_shape = {2, 1, 4},
                  const std::vector<int>& expand_shape = {2, 3, 4},
                  const bool use_shape_tensor = false,
                  const bool use_shape_tensor_list = false) {
  std::string alias{"def"};
  std::unique_ptr<arena::TestCase> tester(
      new ExpandV2ComputeTester<T>(place,
                                   alias,
                                   DDim(x_shape),
                                   expand_shape,
                                   use_shape_tensor,
                                   use_shape_tensor_list));
  arena::Arena arena(std::move(tester), place, abs_error);
  arena.TestPrecision();
}

TEST(ExpandV2, precision) {
  Place place;
  float abs_error = 3e-2;
#if defined(LITE_WITH_NNADAPTER)
  place = TARGET(kNNAdapter);
#if defined(NNADAPTER_WITH_HUAWEI_ASCEND_NPU)
  TestExpandV2<float>(place, abs_error);
  TestExpandV2<float>(place, abs_error, {1, 1, 1}, {2, 3, 4});
  TestExpandV2<float>(place, abs_error, {2, 1, 4}, {2, 3, 4}, true);
  TestExpandV2<float>(place, abs_error, {2, 1, 4}, {2, 2, 3, 4});
  return;
#elif defined(NNADAPTER_WITH_CAMBRICON_MLU)
  abs_error = 1e-5;
  TestExpandV2<float>(place, abs_error);
  TestExpandV2<float>(place, abs_error, {1, 1, 1}, {2, 3, 4});
  TestExpandV2<float>(place, abs_error, {2, 1, 4}, {2, 3, 4}, true);
  TestExpandV2<float>(place, abs_error, {2, 1, 4}, {2, 2, 3, 4});
  return;
#elif defined(NNADAPTER_WITH_INTEL_OPENVINO)
  abs_error = 1e-5;
  TestExpandV2<float>(place, abs_error);
  TestExpandV2<float>(place, abs_error, {1, 1, 1}, {2, 3, 4});
  TestExpandV2<float>(place, abs_error, {2, 1, 4}, {2, 3, 4}, true);
  TestExpandV2<float>(place, abs_error, {2, 1, 4}, {2, 2, 3, 4});
  return;
#else
  return;
#endif
#elif defined(LITE_WITH_XPU)
  place = TARGET(kXPU);
#else
  return;
#endif

  TestExpandV2<float>(place, abs_error);
  TestExpandV2<float>(place, abs_error, {1, 1, 1}, {2, 3, 4});
  TestExpandV2<float>(place, abs_error, {2, 1, 4}, {2, 3, 4}, true);
  TestExpandV2<float>(place, abs_error, {2, 1, 4}, {2, 3, 4}, false, true);
  TestExpandV2<float>(place, abs_error, {2, 1, 4}, {-1, 3, 4});
  TestExpandV2<float>(place, abs_error, {2, 1, 4}, {2, 2, 3, 4});
  TestExpandV2<float>(place, abs_error, {2, 1, 4}, {2, -1, 3, 4});
}

}  // namespace lite
}  // namespace paddle
