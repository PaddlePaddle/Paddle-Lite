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

class Index_selectComputeTester : public arena::TestCase {
 protected:
  std::string input_ = "x";
  std::string index_ = "index";
  std::string output_ = "out";
  std::string alias_ = "fp32";
  int dim_ = 0;              // the axis we choose
  DDim dims_{{5, 5, 5, 5}};  // input tensor dim
  DDim indexdims_{{3}};      // the index index tensor's  dim

 public:
  Index_selectComputeTester(const Place& place,
                            const std::string& alias,
                            int n,
                            int c,
                            int h,
                            int w,
                            int dim)
      : TestCase(place, alias), alias_(alias) {
    dims_ = DDim(std::vector<int64_t>({n, c, h, w}));
    dim_ = dim;
  }
  template <typename indtype>
  void RunBaselineKernel(Scope* scope) {
    auto* out = scope->NewTensor(output_);
    CHECK(out);
    int64_t nchw[] = {dims_[0], dims_[1], dims_[2], dims_[3]};
    std::vector<int64_t> output_shape;
    for (int64_t i = 0; i < dim_; i++) output_shape.push_back(nchw[i]);
    output_shape.push_back(indexdims_[0]);
    for (int64_t i = dim_ + 1; i < 4; i++) output_shape.push_back(nchw[i]);
    DDim output_dims(output_shape);
    out->Resize(output_dims);

    auto* output_data = out->mutable_data<indtype>();
    auto* x = scope->FindTensor(input_);
    const auto* x_data = x->data<indtype>();
    auto* index = scope->FindTensor(index_);
    const auto* index_data = index->data<int64_t>();

    auto x_ddim = x->dims();
    int left = x_ddim.count(0, dim_);
    int middle = x_ddim[dim_];
    int right = x_ddim.count(dim_ + 1, x_ddim.size());

    for (int i = 0; i < left; i++)
      for (int j = 0; j < right; j++)
        for (int k = 0; k < indexdims_[0]; k++)
          output_data[i * indexdims_[0] * right + k * right + j] =
              x_data[i * middle * right + index_data[k] * right + j];
  }

  void RunBaseline(Scope* scope) override {
    if (alias_ == "fp32") {
      RunBaselineKernel<float>(scope);
    }
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) override {
    op_desc->SetType("index_select");
    op_desc->SetInput("X", {input_});
    op_desc->SetInput("Index", {index_});
    op_desc->SetOutput("Out", {output_});
    op_desc->SetAttr("dim", dim_);
  }
  void PrepareData() override {
    std::vector<int64_t> index_data(indexdims_.production());
    for (int i = 0; i < indexdims_.production(); i++) {
      index_data[i] = (i * 997) % dims_[dim_];
    }
    SetCommonTensor(index_, indexdims_, index_data.data());

    if (alias_ == "fp32") {
      std::vector<float> input_data(dims_.production());
      for (int i = 0; i < dims_.production(); i++) {
        float sign = i % 3 == 0 ? -1.0f : 1.0f;
        input_data[i] = sign * static_cast<float>(i % 128) * 0.13f + 1.0f;
      }
      SetCommonTensor(input_, dims_, input_data.data());
    }
  }
};

void TestIndex_select(const Place& place) {
  for (int n : {5}) {
    for (int c : {5}) {
      for (int h : {5}) {
        for (int w : {5}) {
          for (int dim : {0, 1, 2, 3}) {
            std::vector<std::string> alias_vec{"fp32"};
            for (std::string alias : alias_vec) {
              std::unique_ptr<arena::TestCase> tester(
                  new Index_selectComputeTester(place, alias, n, c, h, w, dim));
              arena::Arena arena(std::move(tester), place, 0.00001);
              arena.TestPrecision();
            }
          }
        }
      }
    }
  }
}

TEST(Index_select, precision) {
  Place place;
#if defined(LITE_WITH_ARM)
  place = TARGET(kHost);
#elif defined(LITE_WITH_X86)
  place = TARGET(kHost);
#else
  return;
#endif
  TestIndex_select(place);
}

}  // namespace lite
}  // namespace paddle
