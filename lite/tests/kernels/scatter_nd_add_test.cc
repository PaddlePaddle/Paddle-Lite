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
#include "lite/api/paddle_use_kernels.h"
#include "lite/api/paddle_use_ops.h"
#include "lite/core/test/arena/framework.h"
#include "lite/tests/utils/fill_data.h"

namespace paddle {
namespace lite {

template <typename T, typename IndexType>
void ScatterNdAdd(const IndexType* indexs,
                  const T* updates,
                  T* dst,
                  std::vector<int> x_dims_offset,
                  int index_size,
                  int index_count,
                  int add_size) {
  int index_offset = index_size / index_count;
  for (int i = 0; i < index_count; i++) {
    int dst_offset = 0;
    for (int j = 0; j < index_offset; j++) {
      dst_offset += indexs[j] * x_dims_offset[j];
    }
    indexs += index_offset;
    T* dst_tmp = dst + dst_offset;
    for (int j = 0; j < add_size; j++) {
      dst_tmp[j] += updates[j];
    }
    updates += add_size;
  }
}

template <typename T, typename IndexType>
class ScatterNdAddTester : public arena::TestCase {
 protected:
  std::string x_ = "X";
  std::string index_ = "index";
  std::string updates_ = "updates";
  std::string out_ = "Out";
  DDim x_dims_{{2, 3, 4}};
  DDim index_dims_{{2, 2}};
  DDim updates_dims_{{2, 4}};
  std::vector<IndexType> index_data_{0, 1, 1, 2};

 public:
  ScatterNdAddTester(const Place& place,
                     const std::string& alias,
                     const std::vector<int64_t>& x_shape = {2, 3, 4},
                     const std::vector<int64_t>& index_shape = {2, 2},
                     const std::vector<int64_t>& updates_shape = {2, 4},
                     const std::vector<IndexType>& index_data = {0, 1, 1, 2})
      : TestCase(place, alias),
        x_dims_(DDim(x_shape)),
        index_dims_(DDim(index_shape)),
        updates_dims_(DDim(updates_shape)),
        index_data_(index_data) {}

  void RunBaseline(Scope* scope) override {
    auto* x = scope->FindTensor(x_);
    auto* index = scope->FindTensor(index_);
    auto* updates = scope->FindTensor(updates_);
    auto* out = scope->NewTensor(out_);
    out->CopyDataFrom(*x);

    int index_size = static_cast<int>(index_dims_.production());
    int index_count = index_dims_.count(0, index_dims_.size() - 1);
    int index_step = index_size / index_count;

    std::vector<int> x_dims_offset(x_dims_.size());
    x_dims_offset[x_dims_offset.size() - 1] = 1;
    for (int i = static_cast<int>(x_dims_.size()) - 2; i >= 0; i--) {
      x_dims_offset[i] = x_dims_offset[i + 1] * x_dims_[i + 1];
    }

    int add_size = x_dims_.count(index_step, x_dims_.size());

    auto* indexs_data = index->template data<IndexType>();
    auto* updates_data = updates->template data<T>();
    auto* output_data = out->template mutable_data<T>();
    ScatterNdAdd(indexs_data,
                 updates_data,
                 output_data,
                 x_dims_offset,
                 index_size,
                 index_count,
                 add_size);
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType("scatter_nd_add");
    op_desc->SetInput("X", {x_});
    op_desc->SetInput("Index", {index_});
    op_desc->SetInput("Updates", {updates_});
    op_desc->SetOutput("Out", {out_});
  }

  void PrepareData() override {
    std::vector<T> x_data(x_dims_.production());
    fill_data_rand(x_data.data(),
                   static_cast<T>(-10),
                   static_cast<T>(10),
                   x_dims_.production());
    SetCommonTensor(x_, x_dims_, x_data.data());

    SetCommonTensor(index_, index_dims_, index_data_.data());

    std::vector<T> updates_data(updates_dims_.production());
    fill_data_rand(updates_data.data(),
                   static_cast<T>(-10),
                   static_cast<T>(10),
                   updates_dims_.production());
    SetCommonTensor(updates_, updates_dims_, updates_data.data());
  }
};

template <typename T, typename IndexType>
void TestScatterNdAdd(const Place& place,
                      const float abs_error,
                      const std::string alias) {
  std::unique_ptr<arena::TestCase> tester0(
      new ScatterNdAddTester<T, IndexType>(place, alias));
  arena::Arena arena0(std::move(tester0), place, abs_error);
  arena0.TestPrecision();

  std::unique_ptr<arena::TestCase> tester1(
      new ScatterNdAddTester<T, IndexType>(place,
                                           alias,
                                           std::vector<int64_t>{6, 1},
                                           std::vector<int64_t>{4, 1},
                                           std::vector<int64_t>{4, 1},
                                           std::vector<IndexType>{1, 2, 3, 1}));
  arena::Arena arena1(std::move(tester1), place, abs_error);
  arena1.TestPrecision();

  std::unique_ptr<arena::TestCase> tester2(
      new ScatterNdAddTester<T, IndexType>(place,
                                           alias,
                                           std::vector<int64_t>{2, 2},
                                           std::vector<int64_t>{2, 0},
                                           std::vector<int64_t>{2, 2, 2}));
  arena::Arena arena2(std::move(tester2), place, abs_error);
  arena2.TestPrecision();
}

TEST(scatter_nd_add, precision) {
  Place place;
  float abs_error = 1e-5;
#if defined(LITE_WITH_ARM) || defined(LITE_WITH_X86)
  place = TARGET(kHost);
#else
  return;
#endif

  TestScatterNdAdd<float, int>(place, abs_error, "def");
  TestScatterNdAdd<float, int64_t>(place, abs_error, "float32_int64");
  TestScatterNdAdd<int, int>(place, abs_error, "int32_int32");
  TestScatterNdAdd<int, int64_t>(place, abs_error, "int32_int64");
  TestScatterNdAdd<int64_t, int>(place, abs_error, "int64_int32");
  TestScatterNdAdd<int64_t, int64_t>(place, abs_error, "int64_int64");
}

}  // namespace lite
}  // namespace paddle
