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

namespace paddle {
namespace lite {

class NormalizeComputeTester : public arena::TestCase {
 protected:
  // common attributes for this op.
  std::string input_ = "X";
  std::string output_ = "Out";
  DDim dims_{{10, 20}};
  bool across_spatial_{true};
  int p_{2};
  float eps_{1e-6f};

 public:
  NormalizeComputeTester(const Place& place,
                         const std::string& alias,
                         bool across_spatial,
                         int p,
                         float eps)
      : TestCase(place, alias),
        across_spatial_(across_spatial),
        p_(p),
        eps_(eps) {}

  void RunBaseline(Scope* scope) override {
    LOG(INFO) << "into base line";
    auto* out = scope->NewTensor(output_);
    CHECK(out);
    out->Resize(dims_);
    auto* dst_ptr = out->mutable_data<float>();
    auto* x = scope->FindTensor(input_);
    const auto* src_ptr = x->data<float>();

    int p = p_;
    float eps = eps_;
    bool across_spatial = across_spatial_;
    int n = dims_[0];
    int c = dims_[1];
    int h = dims_[2];
    int w = dims_[3];
    LOG(INFO) << "into spatial";
    if (across_spatial) {
      int compute_size = h * w * c;
      int outer_size = n * c * h * w / compute_size;

      for (int i = 0; i < outer_size; ++i) {
        float sum = 0;

        for (int j = 0; j < compute_size; ++j) {
          if (p == 1) {
            sum += fabsf(src_ptr[j]);
          } else {
            sum += src_ptr[j] * src_ptr[j];
          }
        }

        // LOG(INFO) << "idx: " << i << ", " << "norm: " << sum;

        if (p == 1) {
          sum = 1 / (sum + eps);
        } else {
          sum = 1 / sqrtf(sum + eps);
        }

        for (int j = 0; j < compute_size; ++j) {
          dst_ptr[j] = src_ptr[j] * sum;
        }

        src_ptr += compute_size;
        dst_ptr += compute_size;
      }
    } else {
      LOG(INFO) << "spatial is false";
      int channel_in_size = h * w;

      for (int i = 0; i < n; ++i) {
        const float* src_batch_ptr = src_ptr + i * c * h * w;
        float* dst_batch_ptr = dst_ptr + i * c * h * w;
        LOG(INFO) << "1";
        for (int j = 0; j < h; ++j) {
          for (int k = 0; k < w; ++k) {
            const float* src_pixel = src_batch_ptr + j * w + k;
            float* dst_pixel = dst_batch_ptr + j * w + k;
            float norm = 0.f;
            // LOG(INFO)<<"c:"<<c;

            for (int l = 0; l < c; ++l) {
              if (p == 1) {
                norm += fabsf(src_pixel[l * channel_in_size]);
              } else {
                norm += src_pixel[l * channel_in_size] *
                        src_pixel[l * channel_in_size];
              }
            }
            // LOG(INFO)<<"norm:"<<norm;
            if (p == 1) {
              norm = 1.f / (norm + eps);
            } else {
              norm = 1.f / sqrtf(norm + eps);
            }

            for (int l = 0; l < c; ++l) {
              dst_pixel[l * channel_in_size] =
                  src_pixel[l * channel_in_size] * norm;
              // LOG(INFO)<<"dst:"<<dst_pixel[l * channel_in_size];
              // LOG(INFO)<<"src:"<<src_pixel[l * channel_in_size];
              // LOG(INFO)<<"norm_dd:"<<norm;
            }
          }
        }
      }
    }
    LOG(INFO) << "get out of base Line";
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType("normalize");
    op_desc->SetInput("X", {input_});
    op_desc->SetOutput("Out", {output_});
    op_desc->SetAttr("across_spatial", across_spatial_);
    op_desc->SetAttr("p", p_);
    op_desc->SetAttr("eps", eps_);
  }

  void PrepareData() override {
    std::vector<float> data(dims_.production());

    for (int i = 0; i < dims_.production(); i++) {
      data[i] = i * 1.1;
    }
    SetCommonTensor(input_, dims_, data.data());
  }
};

TEST(Normalize, precision) {
#ifdef LITE_WITH_X86
  Place place(TARGET(kX86));
#endif
#ifdef LITE_WITH_ARM
  Place place(TARGET(kARM));
#endif
  bool across_spatial = false;
  float eps = 1e-6f;
  for (int p : {1, 2}) {
    std::unique_ptr<arena::TestCase> tester(
        new NormalizeComputeTester(place, "def", across_spatial, p, eps));
    arena::Arena arena(std::move(tester), place, 2e-5);
    arena.TestPrecision();
  }
}

}  // namespace lite
}  // namespace paddle
