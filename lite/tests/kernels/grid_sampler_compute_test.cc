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

class GridSamplerComputeTest : public arena::TestCase {
 protected:
  // common attributes for this op.
  std::string input_ = "x";
  std::string output_ = "y";
  std::string grid_ = "grid";

  DDim dims_{{4, 5, 19, 19}};
  bool align_corners_ = true;
  std::string mode_ = "bilinear";
  std::string padding_mode_ = "zeros";

 public:
  GridSamplerComputeTest(const Place& place,
                         const std::string& alias,
                         DDim dims,
                         bool align_corners = true,
                         const std::string& mode = "bilinear",
                         const std::string& padding_mode = "zeros")
      : TestCase(place, alias),
        dims_(dims),
        align_corners_(align_corners),
        mode_(mode),
        padding_mode_(padding_mode) {}

  void RunBaseline(Scope* scope) override {
    auto x = scope->FindTensor(input_);
    auto grid = scope->FindTensor(grid_);
    auto out = scope->NewTensor(output_);
    CHECK(out);
    out->Resize(dims_);

    lite::Tensor new_grid_x, new_grid_y;
    new_grid_x.Resize(grid->dims());
    new_grid_y.Resize(grid->dims());
    float* new_grid_data_x = new_grid_x.mutable_data<float>();
    float* new_grid_data_y = new_grid_y.mutable_data<float>();

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
      float* new_grid_data_xn = new_grid_data_x + n * height * width;
      float* new_grid_data_yn = new_grid_data_y + n * height * width;
      for (int c = 0; c < channel; ++c) {
        const float* x_c = x_n + c * spatial_size;
        float* out_c = out_n + c * spatial_size;
        for (int s = 0; s < spatial_size; ++s) {
          float x = grid_n[s * 2];
          float y = grid_n[s * 2 + 1];
          float xwf = align_corners_ ? (x + 1.f) * 0.5 * (width - 1)
                                     : (x + 1.f) * 0.5 * width - 0.5;
          float ynf = align_corners_ ? (y + 1.f) * 0.5 * (height - 1)
                                     : (y + 1.f) * 0.5 * height - 0.5;

          // clip
          if (padding_mode_ == "zeros") {
            // nothing to do
          } else if (padding_mode_ == "border") {
            xwf = fmin(fmax(xwf, 0), width - 1);
            ynf = fmin(fmax(ynf, 0), height - 1);
          } else if (padding_mode_ == "reflection") {
            if (align_corners_) {
              // x
              float double_range_x = (width - 1) * 2;
              float grid_x_abs = std::abs(xwf);
              float extra_x = grid_x_abs -
                              static_cast<int>(grid_x_abs / double_range_x) *
                                  double_range_x;
              xwf = fmin(extra_x, double_range_x - extra_x);
              // y
              float double_range_y = (height - 1) * 2;
              float grid_y_abs = std::abs(ynf);
              float extra_y = grid_y_abs -
                              static_cast<int>(grid_y_abs / double_range_y) *
                                  double_range_y;
              ynf = fmin(extra_y, double_range_y - extra_y);
            } else {
              // x
              float double_range_x = (width - 1 + 1) * 2;
              float grid_x_abs = std::abs(xwf + 0.5);
              float extra_x = grid_x_abs -
                              static_cast<int>(grid_x_abs / double_range_x) *
                                  double_range_x;
              xwf = fmin(extra_x, double_range_x - extra_x) - 0.5;
              xwf = fmin(fmax(xwf, 0), width - 1);
              // y
              float double_range_y = (height - 1 + 1) * 2;
              float grid_y_abs = std::abs(ynf + 0.5);
              float extra_y = grid_y_abs -
                              static_cast<int>(grid_y_abs / double_range_y) *
                                  double_range_y;
              ynf = fmin(extra_y, double_range_y - extra_y) - 0.5;
              ynf = fmin(fmax(ynf, 0), height - 1);
            }
          } else {
            LOG(FATAL) << "unsupported padding_mode:" << padding_mode_;
          }

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

          if (mode_ == "bilinear") {
            out_c[s] =
                wn * de * ds + en * dw * ds + ws * de * dn + es * dw * dn;
          } else if (mode_ == "nearest") {
            new_grid_data_xn[s] = round(xwf);
            new_grid_data_yn[s] = round(ynf);
          } else {
            LOG(FATAL) << "unsupported mode " << mode_;
          }
        }
      }
    }

    if (mode_ == "bilinear") {
      // nothing to do
    } else if (mode_ == "nearest") {
      auto out_h = grid->dims()[1];
      auto out_w = grid->dims()[2];
      for (int n = 0; n < num; n++) {
        const float* new_grid_data_xn = new_grid_data_x + n * height * width;
        const float* new_grid_data_yn = new_grid_data_y + n * height * width;
        for (int k = 0; k < out_h; k++) {
          for (int l = 0; l < out_w; l++) {
            const float* x = new_grid_data_xn + k * out_w + l;
            const float* y = new_grid_data_yn + k * out_w + l;
            if (inbound(*x, *y, width - 1, height - 1)) {
              for (int j = 0; j < channel; j++) {
                int in_ind_h = round(*y);
                int in_ind_w = round(*x);
                int ind_base = n * channel * out_h * out_w + j * out_h * out_w;
                out_data[ind_base + k * out_w + l] =
                    x_data[ind_base + in_ind_h * width + in_ind_w];
              }
            }
          }
        }
      }
    } else {
      LOG(FATAL) << "unsupported mode " << mode_;
    }
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType("grid_sampler");
    op_desc->SetInput("X", {input_});
    op_desc->SetInput("Grid", {grid_});
    op_desc->SetOutput("Output", {output_});

    op_desc->SetAttr("mode", mode_);
    op_desc->SetAttr("padding_mode", padding_mode_);
    op_desc->SetAttr("align_corners", align_corners_);
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
  for (const bool align_corners : {true, false}) {
    for (const std::string& mode : {"bilinear", "nearest"}) {
      for (const std::string& padding_mode :
           {"zeros", "border", "reflection"}) {
        for (auto& n : {1, 13}) {
          for (auto& c : {1, 3, 8}) {
            for (auto& h : {1, 3, 8, 64}) {
              for (auto& w : {2, 4, 9, 63}) {
                DDim dim_in({n, c, h, w});
                std::unique_ptr<arena::TestCase> tester(
                    new GridSamplerComputeTest(place,
                                               "def",
                                               dim_in,
                                               align_corners,
                                               mode,
                                               padding_mode));
#ifdef LITE_WITH_ARM
                auto& ctx = tester->context()->As<ARMContext>();
                ctx.SetRunMode(lite_api::LITE_POWER_HIGH, 1);
#endif
#ifdef LITE_WITH_X86
                if (padding_mode == "reflection" || mode == "nearest") continue;
#endif
                arena::Arena arena(std::move(tester), place, 6e-5);
                VLOG(5) << "run n: " << n << ", c: " << c << ", h: " << h
                        << ", w: " << w << ", align_corners:" << align_corners
                        << ", mode:" << mode
                        << ", padding_mode:" << padding_mode;
                if (!arena.TestPrecision()) {
                  LOG(ERROR) << "No Pass!!";
                  return;
                }
                // if you want to test this op performance, uncomment the
                // following line
                // arena.TestPerformance();
              }
            }
          }
        }
      }
    }
  }
}

TEST(GridSampler, precision) {
  Place place;
#if defined(LITE_WITH_ARM)
  place = TARGET(kARM);
#elif defined(LITE_WITH_X86)
  place = TARGET(kX86);
#else
  return;
#endif

  test_grid_sampler(place);
}

}  // namespace lite
}  // namespace paddle
