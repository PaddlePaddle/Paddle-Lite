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

template <class XType, class IndexType>
class GatherNdComputeTest : public arena::TestCase {
 protected:
  std::string op_type_ = "gather_nd";
  std::string x_ = "x";
  std::string index_ = "index";
  std::string out_ = "out";
  DDim x_dims_;
  DDim index_dims_;
  std::vector<IndexType> index_data_;

 public:
  GatherNdComputeTest(const Place& place,
                      const std::string& alias,
                      const DDim& x_dims,
                      const DDim& index_dims,
                      const std::vector<IndexType>& index_data)
      : TestCase(place, alias),
        x_dims_(x_dims),
        index_dims_(index_dims),
        index_data_(index_data) {}

  void RunBaseline(Scope* scope) override {
    auto x = scope->FindTensor(x_);
    auto index = scope->FindTensor(index_);
    const auto* x_data = x->template data<XType>();
    const auto* index_data = index->template data<IndexType>();

    auto out = scope->NewTensor(out_);
    auto index_dims_size = index_dims_.size();
    std::vector<int64_t> out_shape;
    for (size_t i = 0; i < index_dims_size - 1; i++) {
      out_shape.emplace_back(index_dims_[i]);
    }
    auto x_dims_size = x_dims_.size();
    for (size_t i = index_dims_[index_dims_size - 1]; i < x_dims_size; i++) {
      out_shape.emplace_back(x_dims_[i]);
    }
    out->Resize(out_shape);
    out->set_lod(x->lod());
    auto out_data = out->template mutable_data<XType>();

    int64_t end_size = index_dims_[index_dims_size - 1];
    auto remain_ddim = index_dims_.Slice(0, index_dims_size - 1);
    int64_t remain_numel = remain_ddim.production();
    int64_t slice_size = 1;
    for (size_t i = end_size; i < x_dims_size; ++i) {
      slice_size *= x_dims_[i];
    }
    const size_t slice_bytes = slice_size * sizeof(XType);

    for (int64_t i = 0; i < remain_numel; ++i) {
      int64_t index_ = 0;
      int64_t temp = 1;
      for (int64_t j = end_size - 1; j >= 0; --j) {
        IndexType index_value = index_data[i * end_size + j];
        index_ += (index_value * temp);
        temp *= x_dims_[j];
      }
      memcpy(
          out_data + i * slice_size, x_data + index_ * slice_size, slice_bytes);
    }
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType(op_type_);
    op_desc->SetInput("X", {x_});
    op_desc->SetInput("Index", {index_});
    op_desc->SetOutput("Out", {out_});
  }

  void PrepareData() override {
    int64_t x_size = x_dims_.production();
    std::vector<XType> x(x_size);
    fill_data_rand(x.data(),
                   static_cast<XType>(-x_size),
                   static_cast<XType>(x_size),
                   x_size);
    SetCommonTensor(x_, x_dims_, x.data());

    SetCommonTensor(index_, index_dims_, index_data_.data());
  }
};

template <class XType, class IndexType>
void TestGatherNdHelper(Place place,
                        float abs_error,
                        const std::vector<int64_t>& x_shape,
                        const std::vector<int64_t>& index_shape,
                        const std::vector<IndexType>& index_data) {
  std::string alias("def");
  std::unique_ptr<arena::TestCase> tester(
      new GatherNdComputeTest<XType, IndexType>(
          place, alias, DDim(x_shape), DDim(index_shape), index_data));
  arena::Arena arena(std::move(tester), place, abs_error);
  arena.TestPrecision();
}

template <class XType, class IndexType>
void TestGatherNd(Place place, float abs_error) {
  std::vector<int64_t> x_shape{2, 3, 4};
  TestGatherNdHelper<XType, IndexType>(place, abs_error, x_shape, {1, 1}, {1});
  TestGatherNdHelper<XType, IndexType>(
      place, abs_error, x_shape, {1, 2}, {0, 2});
  TestGatherNdHelper<XType, IndexType>(
      place, abs_error, x_shape, {1, 3}, {1, 2, 3});
  TestGatherNdHelper<XType, IndexType>(
      place, abs_error, x_shape, {2, 2}, {1, 2, 0, 1});
}

TEST(gather_nd, precision) {
  float abs_error = 1e-5;
  Place place;
#if defined(LITE_WITH_ARM) || defined(LITE_WITH_X86)
  place = Place(TARGET(kHost), PRECISION(kAny));
#else
  return;
#endif

  TestGatherNd<float, int32_t>(place, abs_error);
  TestGatherNd<float, int64_t>(place, abs_error);
}

}  // namespace lite
}  // namespace paddle
