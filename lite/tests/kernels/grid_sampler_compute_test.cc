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
#include "lite/core/arena/framework.h"
#include "lite/tests/utils/fill_data.h"

namespace paddle {
namespace lite {

class GridSamplerComputeTest : public arena::TestCase {
 protected:
  // common attributes for this op.
  std::string input_ = "x";
  std::string output_ = "y";
  std::string grid_ = "grid";

  DDim dims_{{4, 5, 19, 19}};

 public:
  GridSamplerComputeTest(const Place& place,
                         const std::string& alias,
                         DDim dims)
      : TestCase(place, alias), dims_(dims) {}

  void RunBaseline(Scope* scope) override {
    auto x = scope->FindTensor(input_);
    auto grid = scope->FindTensor(grid_);
    auto out = scope->NewTensor(output_);
    CHECK(out);
    out->Resize(dims_);

    const float* x_data = x->data<float>();
    const float* grid_data = grid->data<float>();
    float* out_data = out->mutable_data<float>();

    int num = x->dims()[0];
    int channel = x->dims()[1];
    int height = x->dims()[2];
    int width = x->dims()[3];
    int spatial_size = height * width;

    auto inbound = [](int x, int y, float x_max, float y_max) {
      if (x < 0 || x > x_max || y < 0 || y > y_max) {
        return false;
      }
      return true;
    };

    for (int n = 0; n < num; ++n) {
      const float* x_n = x_data + n * channel * height * width;
      float* out_n = out_data + n * channel * height * width;
      const float* grid_n = grid_data + n * height * width * 2;
      for (int c = 0; c < channel; ++c) {
        const float* x_c = x_n + c * spatial_size;
        float* out_c = out_n + c * spatial_size;
        for (int s = 0; s < spatial_size; ++s) {
          float x = grid_n[s * 2];
          float y = grid_n[s * 2 + 1];
          float xwf = (x + 1.f) * 0.5 * (width - 1);
          float ynf = (y + 1.f) * 0.5 * (height - 1);
          int xw = floor(xwf);
          int xe = xw + 1;
          int yn = floor(ynf);
          int ys = yn + 1;

          float dw = xwf - xw;
          float de = xe - xwf;
          float dn = ynf - yn;
          float ds = ys - ynf;

          float wn = inbound(xw,
                             yn,
                             static_cast<float>(width - 1),
                             static_cast<float>(height - 1))
                         ? x_c[yn * width + xw]
                         : 0.f;
          float en = inbound(xe,
                             yn,
                             static_cast<float>(width - 1),
                             static_cast<float>(height - 1))
                         ? x_c[yn * width + xe]
                         : 0.f;
          float ws = inbound(xw,
                             ys,
                             static_cast<float>(width - 1),
                             static_cast<float>(height - 1))
                         ? x_c[ys * width + xw]
                         : 0.f;
          float es = inbound(xe,
                             ys,
                             static_cast<float>(width - 1),
                             static_cast<float>(height - 1))
                         ? x_c[ys * width + xe]
                         : 0.f;

          out_c[s] = wn * de * ds + en * dw * ds + ws * de * dn + es * dw * dn;
        }
      }
    }
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType("grid_sampler");
    op_desc->SetInput("X", {input_});
    op_desc->SetInput("Grid", {grid_});
    op_desc->SetOutput("Output", {output_});
  }

  void PrepareData() override {
    std::vector<float> din(dims_.production());
    fill_data_rand(din.data(), -1.f, 1.f, dims_.production());

    DDim gird_dims{{dims_[0], dims_[2], dims_[3], 2}};
    std::vector<float> grid(gird_dims.production());
    fill_data_rand(grid.data(), -1.f, 1.f, gird_dims.production());

    SetCommonTensor(input_, dims_, din.data());
    SetCommonTensor(grid_, gird_dims, grid.data());
  }
};

void test_grid_sampler(Place place) {
  for (auto& n : {1, 13}) {
    for (auto& c : {1, 3, 8}) {
      for (auto& h : {1, 3, 8, 64}) {
        for (auto& w : {2, 4, 9, 63}) {
          DDim dim_in({n, c, h, w});
          std::unique_ptr<arena::TestCase> tester(
              new GridSamplerComputeTest(place, "def", dim_in));
#ifdef LITE_WITH_ARM
          auto& ctx = tester->context()->As<ARMContext>();
          ctx.SetRunMode(lite_api::LITE_POWER_HIGH, 1);
#endif
          arena::Arena arena(std::move(tester), place, 6e-5);
          LOG(INFO) << "run n: " << n << ", c: " << c << ", h: " << h
                    << ", w: " << w;
          if (!arena.TestPrecision()) {
            LOG(ERROR) << "No Pass!!";
            return;
          }
          // if you want to test this op performance, uncomment the following
          // line
          // arena.TestPerformance();
        }
      }
    }
  }
}

TEST(GridSampler, precision) {
#ifdef LITE_WITH_ARM
  Place place(TARGET(kARM));
  test_grid_sampler(place);
#endif
}

}  // namespace lite
}  // namespace paddle
