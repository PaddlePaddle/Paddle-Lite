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

template <class T>
class GatherTreeComputeTest : public arena::TestCase {
 protected:
  std::string op_type_ = "gather_tree";
  std::string ids_ = "ids";
  std::string parents_ = "parents";
  std::string out_ = "out";
  DDim ids_dims_;

 public:
  GatherTreeComputeTest(const Place& place,
                        const std::string& alias,
                        const DDim& ids_dims)
      : TestCase(place, alias), ids_dims_(ids_dims) {}

  void RunBaseline(Scope* scope) override {
    auto ids = scope->FindTensor(ids_);
    auto parents = scope->FindTensor(parents_);
    const auto* ids_data = ids->template data<T>();
    const auto* parents_data = parents->template data<T>();
    auto out = scope->NewTensor(out_);
    out->Resize(ids_dims_);
    auto out_data = out->template mutable_data<T>();

    int max_length = ids_dims_[0];
    int batch_size = ids_dims_[1];
    int beam_size = ids_dims_[2];

    for (int batch = 0; batch < batch_size; batch++) {
      for (int beam = 0; beam < beam_size; beam++) {
        auto idx = (max_length - 1) * batch_size * beam_size +
                   batch * beam_size + beam;
        out_data[idx] = ids_data[idx];
        auto parent = parents_data[idx];
        for (int step = max_length - 2; step >= 0; step--) {
          idx = step * batch_size * beam_size + batch * beam_size;
          out_data[idx + beam] = ids_data[idx + parent];
          parent = parents_data[idx + parent];
        }
      }
    }
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType(op_type_);
    op_desc->SetInput("Ids", {ids_});
    op_desc->SetInput("Parents", {parents_});
    op_desc->SetOutput("Out", {out_});
  }

  void PrepareData() override {
    int64_t size = ids_dims_.production();
    std::vector<T> ids(size);
    fill_data_rand(ids.data(), static_cast<T>(0), static_cast<T>(size), size);
    SetCommonTensor(ids_, ids_dims_, ids.data());

    std::vector<T> parents(size);
    fill_data_rand(
        parents.data(), static_cast<T>(0), static_cast<T>(ids_dims_[2]), size);
    SetCommonTensor(parents_, ids_dims_, parents.data());
  }
};

template <class T = int32_t>
void TestGatherTree(Place place,
                    float abs_error,
                    const std::vector<int64_t>& ids_shape) {
  auto precision = lite_api::PrecisionTypeTrait<T>::Type();
  std::string alias("def");
  switch (precision) {
    case lite_api::PrecisionType::kInt32:
      alias = std::string("int32");
      break;
    case lite_api::PrecisionType::kInt64:
      alias = std::string("int64");
      break;
    default:
      LOG(FATAL) << "unsupported precision: "
                 << lite_api::PrecisionToStr(precision);
  }

  std::unique_ptr<arena::TestCase> tester(
      new GatherTreeComputeTest<T>(place, alias, DDim(ids_shape)));
  arena::Arena arena(std::move(tester), place, abs_error);
  arena.TestPrecision();
}

TEST(gather_tree, precision) {
  float abs_error = 1e-5;
  Place place;
#if defined(LITE_WITH_ARM) || defined(LITE_WITH_X86)
  place = TARGET(kHost);
#else
  return;
#endif

  std::vector<std::vector<int64_t>> shapes{{3, 2, 2}, {10, 9, 12}};
  for (auto shape : shapes) {
    TestGatherTree<int32_t>(place, abs_error, shape);
    TestGatherTree<int64_t>(place, abs_error, shape);
  }
}

}  // namespace lite
}  // namespace paddle
