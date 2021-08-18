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

class PolygonBoxTransformComputeTester : public arena::TestCase {
 protected:
  std::string x_ = "X";
  std::string out_ = "Out";
  DDim x_dims_;

 public:
  PolygonBoxTransformComputeTester(const Place& place,
                                   const std::string& alias,
                                   const DDim& x_dims)
      : TestCase(place, alias), x_dims_(x_dims) {}

  void RunBaseline(Scope* scope) override {
    auto* out = scope->NewTensor(out_);
    out->Resize(x_dims_);
    auto* x = scope->FindTensor(x_);
    const float* x_data = x->data<float>();
    float* out_data = out->mutable_data<float>();

    int batch_size = x_dims_[0];
    int geo_channel = x_dims_[1];
    int height = x_dims_[2];
    int width = x_dims_[3];
    int id = 0;
    for (int id_n = 0; id_n < batch_size * geo_channel; ++id_n) {
      for (int id_h = 0; id_h < height; ++id_h) {
        for (int id_w = 0; id_w < width; ++id_w) {
          id = id_n * height * width + width * id_h + id_w;
          if (id_n % 2 == 0) {
            out_data[id] = id_w * 4 - x_data[id];
          } else {
            out_data[id] = id_h * 4 - x_data[id];
          }
        }
      }
    }
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType("polygon_box_transform");
    op_desc->SetInput("Input", {x_});
    op_desc->SetOutput("Output", {out_});
  }

  void PrepareData() override {
    std::vector<float> x_data(x_dims_.production());
    fill_data_rand(x_data.data(), -1.f, 1.f, x_dims_.production());
    SetCommonTensor(x_, x_dims_, x_data.data());
  }
};

TEST(polygon_box_transform, precision) {
  Place place;
  float abs_error = 1e-5;
#if defined(LITE_WITH_ARM) || defined(LITE_WITH_X86)
  place = TARGET(kHost);
#else
  return;
#endif

  for (auto x_shape :
       std::vector<std::vector<int64_t>>{{1, 5, 6, 7}, {10, 5, 8, 2}}) {
    std::unique_ptr<arena::TestCase> tester(
        new PolygonBoxTransformComputeTester(place, "def", DDim(x_shape)));
    arena::Arena arena(std::move(tester), place, abs_error);
    arena.TestPrecision();
  }
}

}  // namespace lite
}  // namespace paddle
