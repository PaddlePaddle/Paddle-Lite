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
  bool has_scale_bias_ = true;

 public:
  GroupNormComputeTest(const Place& place,
                       const std::string& alias,
                       DDim dims,
                       float epsilon,
                       int groups,
                       bool has_scale_bias)
      : TestCase(place, alias),
        dims_(dims),
        epsilon_(epsilon),
        groups_(groups),
        has_scale_bias_(has_scale_bias) {}

  void RunBaseline(Scope* scope) override {
    const Tensor* x = scope->FindTensor(x_);
    const float* x_data = x->data<float>();
    const float* scale_data =
        has_scale_bias_ ? scope->FindTensor(scale_)->data<float>() : nullptr;
    const float* bias_data =
        has_scale_bias_ ? scope->FindTensor(bias_)->data<float>() : nullptr;

    auto y = scope->NewTensor(y_);
    auto saved_mean = scope->NewTensor(saved_mean_);
    auto saved_variance = scope->NewTensor(saved_variance_);
    int n = x->dims()[0];
    int c = x->dims()[1];
    int groups = groups_;
    float epsilon = epsilon_;

    CHECK(y);
    CHECK(saved_mean);
    CHECK(saved_variance);
    CHECK_GT(groups, 0);
    CHECK_LE(groups, c);
    DDim saved_dim({n, groups});
    y->Resize(dims_);
    saved_mean->Resize(saved_dim);
    saved_variance->Resize(saved_dim);

    auto y_data = y->mutable_data<float>();
    auto saved_mean_data = saved_mean->mutable_data<float>();
    auto saved_variance_data = saved_variance->mutable_data<float>();

    int imsize = x->dims()[2] * x->dims()[3];
    const int group_size = (c - 1) / groups + 1;

    // similar to paddle code
    auto* iter_x_data = x_data;
    auto* iter_y_data = y_data;

    for (int bid = 0; bid < n; bid++) {
      for (int gid = 0; gid < groups; gid++) {
        float x_mean = 0, x_var = 0;
        int number =
            std::min(group_size, static_cast<int>(c - gid * group_size));
        auto* tmp_x = iter_x_data;

        for (int cid = 0; cid < number; cid++) {
          for (int imid = 0; imid < imsize; imid++, iter_x_data++) {
            x_mean += iter_x_data[0];
            x_var += iter_x_data[0] * iter_x_data[0];
          }
        }

        x_mean /= number * imsize;
        // x_var /= number * imsize;
        // x_var = x_var - x_mean * x_mean;
        // higher precision than above
        x_var = (x_var - x_mean * x_mean * imsize * number) / (number * imsize);
        float var_inv = 1.0 / sqrt(x_var + epsilon);
        saved_mean_data[bid * groups + gid] = x_mean;
        saved_variance_data[bid * groups + gid] = x_var;

        for (int cid = 0; cid < number; cid++) {
          for (int imid = 0; imid < imsize; imid++, tmp_x++, iter_y_data++) {
            float val = (tmp_x[0] - x_mean) * var_inv;
            if (scale_data) val *= scale_data[gid * group_size + cid];
            if (bias_data) val += bias_data[gid * group_size + cid];
            iter_y_data[0] = val;
          }
        }
      }
    }
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType("group_norm");
    op_desc->SetInput("X", {x_});
    if (has_scale_bias_) {
      op_desc->SetInput("Bias", {bias_});
      op_desc->SetInput("Scale", {scale_});
    }
    op_desc->SetOutput("Y", {y_});
    op_desc->SetOutput("SavedMean", {saved_mean_});
    op_desc->SetOutput("SavedVariance", {saved_variance_});
    op_desc->SetAttr("epsilon", epsilon_);
    op_desc->SetAttr("groups", groups_);
  }

  void PrepareData() override {
    std::vector<float> x(dims_.production());
    fill_data_rand(x.data(), -1.f, 1.f, dims_.production());
    SetCommonTensor(x_, dims_, x.data());

    if (has_scale_bias_) {
      DDim scale_bias_dims{{dims_[1]}};
      std::vector<float> scale(scale_bias_dims.production());
      fill_data_rand(scale.data(), -1.f, 1.f, scale_bias_dims.production());
      std::vector<float> bias(scale_bias_dims.production());
      fill_data_rand(bias.data(), -1.f, 1.f, scale_bias_dims.production());
      SetCommonTensor(scale_, scale_bias_dims, scale.data(), {}, true);
      SetCommonTensor(bias_, scale_bias_dims, bias.data(), {}, true);
    }
  }
};

void TestGroupNorm(Place place,
                   float abs_error = 6e-5,
                   std::vector<std::string> ignored_outs = {}) {
  for (auto& n : {1, 3}) {
    for (auto& c : {1, 8, 32}) {
      for (auto& h : {1, 16}) {
        for (auto& w : {1, 18}) {
          for (auto& has_scale_bias : {true, false}) {
            DDim dim_in({n, c, h, w});
            float epsilon = 1e-3f;
            for (auto& groups : {1, 2, 4, 8}) {
              if (c % groups != 0) {
                continue;
              }
              std::unique_ptr<arena::TestCase> tester(new GroupNormComputeTest(
                  place, "def", dim_in, epsilon, groups, has_scale_bias));
#if defined(NNADAPTER_WITH_HUAWEI_ASCEND_NPU)
              if (w == 1 && h == 1 && (n != 1 || c != 1)) continue;
#endif
#ifdef LITE_WITH_ARM
              if (place == TARGET(kARM)) {
                auto& ctx = tester->context()->As<ARMContext>();
                ctx.SetRunMode(lite_api::LITE_POWER_HIGH, 4);
              }
#endif
              arena::Arena arena(std::move(tester), place, abs_error);
              if (!arena.TestPrecision(ignored_outs)) {
                LOG(ERROR) << "run n: " << n << ", c: " << c << ", h: " << h
                           << ", w: " << w << ", groups: " << groups
                           << ", has_scale_bias:" << has_scale_bias;
                return;
              }
            }
          }
        }
      }
    }
  }
}

TEST(GroupNorm, precision) {
  Place place;
  float abs_error = 3e-3;
  std::vector<std::string> ignored_outs = {};
#if defined(LITE_WITH_NNADAPTER)
  place = TARGET(kNNAdapter);
#if defined(NNADAPTER_WITH_HUAWEI_ASCEND_NPU)
  abs_error = 1e-1;
  ignored_outs = {"saved_mean", "saved_variance"};
#else
  return;
#endif
#elif defined(LITE_WITH_ARM)
  place = TARGET(kARM);
#elif defined(LITE_WITH_X86)
  place = TARGET(kX86);
#else
  return;
#endif
  TestGroupNorm(place, abs_error, ignored_outs);
}

}  // namespace lite
}  // namespace paddle
