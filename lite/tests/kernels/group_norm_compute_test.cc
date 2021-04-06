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

class GroupNormComputeTest : public arena::TestCase {
 protected:
  // common attributes for this op.
  std::string x_ = "x";
  std::string y_ = "y";
  std::string saved_mean_ = "saved_mean";
  std::string saved_variance_ = "saved_variance";
  std::string scale_ = "scale";
  std::string bias_ = "bias";

  DDim dims_{{4, 5, 19, 19}};
  float epsilon_ = 1e-5f;
  int groups_ = 1;
  std::string data_layout_str_ = "NCHW";

 public:
  GroupNormComputeTest(const Place& place,
                       const std::string& alias,
                       DDim dims,
                       float epsilon,
                       int groups,
                       std::string data_layout_str)
      : TestCase(place, alias),
        dims_(dims),
        epsilon_(epsilon),
        groups_(groups),
        data_layout_str_(data_layout_str) {}

  void RunBaseline(Scope* scope) override {
    auto x = scope->FindTensor(x_);
    auto scale = scope->FindTensor(scale_);
    auto bias = scope->FindTensor(bias_);
    auto y = scope->NewTensor(y_);
    auto saved_mean = scope->NewTensor(saved_mean_);
    auto saved_variance = scope->NewTensor(saved_variance_);
    CHECK(y);
    CHECK(saved_mean);
    CHECK(saved_variance);
    DDim saved_dim({dims_[0], groups_});
    y->Resize(dims_);
    saved_mean->Resize(saved_dim);
    saved_variance->Resize(saved_dim);

    auto x_data = x->data<float>();
    auto scale_data = scale->data<float>();
    auto bias_data = bias->data<float>();
    auto y_data = y->mutable_data<float>();
    auto mean_data = saved_mean->mutable_data<float>();
    auto var_data = saved_variance->mutable_data<float>();

    auto x_dims = x->dims();
    int groups = groups_;
    int channels =
        (data_layout_str_ == "NCHW") ? x_dims[1] : x_dims[x_dims.size() - 1];
    int group_size = (channels - 1) / groups + 1;
    int imsize = (data_layout_str_ == "NCHW") ? (x_dims[2] * x_dims[3])
                                              : (x_dims[1] * x_dims[2]);

    auto* iter_x_data = x_data;
    auto* iter_y_data = y_data;
    for (int bid = 0; bid < x_dims[0]; bid++) {
      for (int gid = 0; gid < groups; gid++) {
        float x_mean = 0;
        float x_var = 0;
        int number =
            std::min(group_size, static_cast<int>(channels - gid * group_size));
        auto* tmp_x = iter_x_data;
        auto* x_src_data = iter_x_data;
        auto* tmp_y = iter_y_data;
        auto* y_src_data = iter_y_data;

        if (data_layout_str_ == "NCHW") {
          for (int cid = 0; cid < number; cid++) {
            for (int imid = 0; imid < imsize; imid++, iter_x_data++) {
              x_mean += iter_x_data[0];
              x_var += iter_x_data[0] * iter_x_data[0];
            }
          }
        } else {
          for (int cid = 0; cid < number; cid++) {
            iter_x_data = tmp_x + cid;
            for (int imid = 0; imid < imsize; imid++, iter_x_data += channels) {
              x_mean += iter_x_data[0];
              x_var += iter_x_data[0] * iter_x_data[0];
            }
          }
          iter_x_data = tmp_x + group_size;
        }

        x_mean /= number * imsize;
        x_var /= number * imsize;
        x_var = x_var - x_mean * x_mean;
        float var_inv = 1.0 / std::sqrt(x_var + epsilon_);
        mean_data[bid * groups + gid] = x_mean;
        var_data[bid * groups + gid] = x_var;

        if (data_layout_str_ == "NCHW") {
          for (int cid = 0; cid < number; cid++) {
            for (int imid = 0; imid < imsize; imid++, tmp_x++, iter_y_data++) {
              float val = (tmp_x[0] - x_mean) * var_inv;
              if (scale_data) val *= scale_data[gid * group_size + cid];
              if (bias_data) val += bias_data[gid * group_size + cid];
              iter_y_data[0] = val;
            }
          }
        } else {
          for (int cid = 0; cid < number; cid++) {
            tmp_x = x_src_data + cid;
            iter_y_data = y_src_data + cid;
            for (int imid = 0; imid < imsize;
                 imid++, tmp_x += channels, iter_y_data += channels) {
              float val = (tmp_x[0] - x_mean) * var_inv;
              if (scale_data) val *= scale_data[gid * group_size + cid];
              if (bias_data) val += bias_data[gid * group_size + cid];
              iter_y_data[0] = val;
            }
          }
          iter_y_data = tmp_y + group_size;
        }
      }
      if (data_layout_str_ == "NCHW") {
        iter_x_data = x_data + (bid + 1) * channels * imsize;
        iter_y_data = y_data + (bid + 1) * channels * imsize;
      }
    }
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType("group_norm");
    op_desc->SetInput("X", {x_});
    op_desc->SetInput("Bias", {bias_});
    op_desc->SetInput("Scale", {scale_});
    op_desc->SetOutput("Y", {y_});
    op_desc->SetOutput("Mean", {saved_mean_});
    op_desc->SetOutput("Variance", {saved_variance_});
    op_desc->SetAttr("epsilon", epsilon_);
    op_desc->SetAttr("groups", groups_);
    op_desc->SetAttr("data_layout", data_layout_str_);
  }

  void PrepareData() override {
    std::vector<float> x(dims_.production());
    fill_data_rand(x.data(), -1.f, 1.f, dims_.production());

    DDim scale_bias_dims{{dims_[1]}};
    std::vector<float> scale(scale_bias_dims.production());
    fill_data_rand(scale.data(), -1.f, 1.f, scale_bias_dims.production());
    std::vector<float> bias(scale_bias_dims.production());
    fill_data_rand(bias.data(), -1.f, 1.f, scale_bias_dims.production());

    SetCommonTensor(x_, dims_, x.data());
    SetCommonTensor(scale_, scale_bias_dims, scale.data(), {}, true);
    SetCommonTensor(bias_, scale_bias_dims, bias.data(), {}, true);
  }
};

void TestGroupNorm(Place place,
                   float abs_error = 6e-5,
                   std::vector<std::string> ignored_outs = {}) {
  for (auto& n : {1, 3, 16}) {
    for (auto& c : {1, 2}) {
      for (auto& h : {1, 16, 33, 56}) {
        for (auto& w : {1, 17, 55}) {
          for (auto& groups : {1, 2, 4}) {
            if (c % groups != 0) {
              continue;
            }
            DDim dim_in({n, c, h, w});
            float epsilon = 1e-5f;
            std::unique_ptr<arena::TestCase> tester(new GroupNormComputeTest(
                place, "def", dim_in, epsilon, groups, "NCHW"));
#ifdef LITE_WITH_ARM
            if (place == TARGET(kARM)) {
              auto& ctx = tester->context()->As<ARMContext>();
              ctx.SetRunMode(lite_api::LITE_POWER_HIGH, 4);
            }
#endif
            arena::Arena arena(std::move(tester), place, abs_error);
            if (!arena.TestPrecision(ignored_outs)) {
              LOG(ERROR) << "run n: " << n << ", c: " << c << ", h: " << h
                         << ", w: " << w;
              return;
            }
          }
        }
      }
    }
  }
}

TEST(GroupNorm, precision) {
  Place place;
  float abs_error = 6e-5;
  std::vector<std::string> ignored_outs = {};
#ifdef LITE_WITH_ARM
  place = TARGET(kARM);
#else
  return;
#endif
  TestGroupNorm(place, abs_error, ignored_outs);
}
}  // namespace lite
}  // namespace paddle
