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
          bool has_shape_input = false,
          bool has_expand_shapes_tensor = false>
class ExpandV2ComputeTester : public arena::TestCase {
 protected:
  // common attributes for this op.
  std::string x_ = "X";
  std::string out_ = "Out";
  std::string Shape_ = "Shape";
  std::string shape_ = "shape";
  std::string expand_shapes_tensor_ = "expand_shapes_tensor";
  std::vector<int> input_shape_;
  std::vector<int> expand_shape_;

 public:
  ExpandV2ComputeTester(const Place& place,
                        const std::string& alias,
                        const std::vector<int>& input_shape,
                        const std::vector<int>& expand_shape)
      : TestCase(place, alias),
        input_shape_(input_shape),
        expand_shape_(expand_shape) {}

  void RunBaseline(Scope* scope) override {
    std::vector<int64_t> vec_in_dims;
    for (int i = 0; i < input_shape_.size(); ++i) {
      vec_in_dims.push_back(static_cast<int64_t>(input_shape_[i]));
    }
    auto diff = expand_shape_.size() - vec_in_dims.size();
    vec_in_dims.insert(vec_in_dims.begin(), diff, 1);
    std::vector<int> repeat_times(vec_in_dims.size());
    for (size_t i = 0; i < vec_in_dims.size(); ++i) {
      if (i < diff) {
        repeat_times[i] = expand_shape_[i];
      } else if (expand_shape_[i] > 0) {
        if (vec_in_dims[i] != 1) {
          repeat_times[i] = 1;
        } else {
          repeat_times[i] = expand_shape_[i];
        }
      } else {
        repeat_times[i] = 1;
      }
    }

    const auto* input = scope->FindTensor(x_);
    CHECK(input);
    auto* out = scope->NewTensor(out_);
    std::vector<int64_t> out_dims;
    for (int i = 0; i < expand_shape_.size(); i++) {
      out_dims.push_back(expand_shape_[i]);
    }
    out->Resize(DDim(out_dims));
    CHECK(out);
    const T* src = input->template data<T>();
    T* dst = out->template mutable_data<T>();
    DDim new_in_shape;
    new_in_shape.ConstructFrom(vec_in_dims);
    int dims = repeat_times.size();
    DDim out_shape = out->dims();
    int inner_num = 1;
    int index = dims - 1;
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
    for (int index = dims - 2; index >= 0; --index) {
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
    if (has_shape_input) {
      op_desc->SetInput("Shape", {Shape_});
    }
    op_desc->SetOutput("Out", {out_});
    op_desc->SetAttr("shape", expand_shape_);
  }

  void PrepareData() override {
    std::vector<int64_t> input_shape;
    for (int i = 0; i < input_shape_.size(); i++) {
      input_shape.push_back(input_shape_[i]);
    }
    auto input_dims = DDim(input_shape);
    std::vector<T> in_data(input_dims.production());
    for (int i = 0; i < input_dims.production(); ++i) {
      in_data[i] = i;
    }
    SetCommonTensor(x_, input_dims, in_data.data());
    if (has_shape_input) {
      SetCommonTensor(Shape_,
                      DDim{{static_cast<int64_t>(expand_shape_.size())}},
                      expand_shape_.data(),
                      {},
                      true);
    }
  }
};

template <class T,
          bool has_shape_input = false,
          bool has_expand_times_tensor = false>
void test_expand_v2(Place place,
                    float abs_error,
                    std::vector<int> input_shape,
                    std::vector<int> expand_shape) {
  std::string alias{"def"};
  std::unique_ptr<arena::TestCase> tester(
      new ExpandV2ComputeTester<T, has_shape_input, has_expand_times_tensor>(
          place, alias, input_shape, expand_shape));
  arena::Arena arena(std::move(tester), place, abs_error);
  arena.TestPrecision();
}

#if defined(LITE_WITH_NNADAPTER) && defined(NNADAPTER_WITH_HUAWEI_ASCEND_NPU)
TEST(ExpandV2, precision) {
  Place place = TARGET(kNNAdapter);
  float abs_error = 3e-2;
  test_expand_v2<float>(
      place, abs_error, std::vector<int>({3, 1}), std::vector<int>({3, 4}));
  test_expand_v2<float, true>(
      place, abs_error, std::vector<int>({3, 1}), std::vector<int>({3, 4}));
  test_expand_v2<float>(
      place, abs_error, std::vector<int>({3, 1}), std::vector<int>({3, 3, 4}));
  test_expand_v2<float, true>(
      place, abs_error, std::vector<int>({3, 1}), std::vector<int>({3, 3, 4}));
}
#endif

}  // namespace lite
}  // namespace paddle
