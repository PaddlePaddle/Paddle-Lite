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

void scatter_basic(const int64_t* indexs,
                   const float* src,
                   float* dst,
                   int index_size,
                   int num,
                   int size,
                   bool overwrite) {
  memset(reinterpret_cast<char*>(dst), 0, sizeof(float) * size * num);
  if (overwrite) {
    for (int i = 0; i < index_size; i++) {
      const float* din = src + i * size;
      memcpy(dst + indexs[i] * size, din, sizeof(float) * size);
    }
  } else {
    for (int i = 0; i < index_size; i++) {
      const float* din = src + i * size;
      float* dout = dst + indexs[i] * size;
      for (int j = 0; j < size; j++) {
        dout[j] += din[j];
      }
    }
  }
}

class ScatterComputeTester : public arena::TestCase {
 protected:
  // common attributes for this op.
  std::string input_ = "x";
  std::string indexs_ = "indexs";
  std::string updates_ = "updates";
  std::string output_ = "out";
  DDim up_dims_{{1}};
  DDim id_dims_{{1}};
  DDim x_dims_{{1}};
  int index_size_ = 0;
  bool overwrite_ = false;

 public:
  ScatterComputeTester(const Place& place,
                       const std::string& alias,
                       DDim up_dims,
                       DDim id_dims,
                       DDim x_dims,
                       bool overwrite,
                       int index_size)
      : TestCase(place, alias),
        up_dims_(up_dims),
        id_dims_(id_dims),
        x_dims_(x_dims),
        index_size_(index_size),
        overwrite_(overwrite) {}

  void RunBaseline(Scope* scope) override {
    auto* indexs_t = scope->FindMutableTensor(indexs_);
    auto* updates_t = scope->FindMutableTensor(updates_);
    const auto* indexs_data = indexs_t->data<int64_t>();
    const auto* updates_data = updates_t->data<float>();
    auto* out = scope->NewTensor(output_);

    out->Resize(x_dims_);

    auto* out_data = out->mutable_data<float>();
    int in_n = x_dims_[0];
    int in_c = x_dims_[1];
    int in_h = x_dims_[2];
    int in_w = x_dims_[3];
    int size = in_c * in_h * in_w;

    scatter_basic(indexs_data,
                  updates_data,
                  out_data,
                  index_size_,
                  in_n,
                  size,
                  overwrite_);
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType("scatter");
    op_desc->SetInput("X", {input_});
    op_desc->SetInput("Ids", {indexs_});
    op_desc->SetInput("Updates", {updates_});
    op_desc->SetOutput("Out", {output_});
    op_desc->SetAttr("overwrite", overwrite_);
  }

  void PrepareData() override {
    std::vector<float> data(x_dims_.production());
    for (int i = 0; i < x_dims_.production(); i++) {
      data[i] = i * 1.0;
    }
    SetCommonTensor(input_, x_dims_, data.data());
    std::vector<float> update(up_dims_.production());
    for (int i = 0; i < up_dims_.production(); i++) {
      update[i] = i * 1.0;
    }
    SetCommonTensor(updates_, up_dims_, update.data());
    std::vector<int64_t> index(id_dims_.production());
    for (int i = 0; i < id_dims_.production(); i++) {
      index[i] = i;
    }
    SetCommonTensor(indexs_, id_dims_, index.data());
  }
};

void test_scatter(Place place) {
  for (auto n : {1, 3}) {
    for (auto c : {1, 2}) {
      for (auto h : {1, 3}) {
        for (auto w : {1, 3}) {
          for (bool overwrite : {false, true}) {
            auto x_dims = DDim(std::vector<int64_t>({n, c, h, w}));
            auto up_dims = DDim(std::vector<int64_t>({n, c, h, w}));
            auto id_dims = DDim(std::vector<int64_t>({n}));
            std::unique_ptr<arena::TestCase> tester(new ScatterComputeTester(
                place, "ids_int64", up_dims, id_dims, x_dims, overwrite, n));
            arena::Arena arena(std::move(tester), place, 2e-5);
            arena.TestPrecision();
          }
        }
      }
    }
  }
}

TEST(Scatter, precision) {
#ifdef LITE_WITH_ARM
  Place place(TARGET(kARM));
  test_scatter(place);
#endif
}

}  // namespace lite
}  // namespace paddle
