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

class ArgmaxComputeTester : public arena::TestCase {
 protected:
  // common attributes for this op.
  std::string input_ = "x";
  std::string output_ = "out";
  int axis_ = 0.;
  DDim dims_{{2, 5, 20, 30}};

 public:
  ArgmaxComputeTester(const Place& place,
                      const std::string& alias,
                      int axis,
                      int n,
                      int c,
                      int h,
                      int w)
      : TestCase(place, alias), axis_(axis) {
    dims_ = DDim(std::vector<int64_t>({n, c, h, w}));
  }

  void RunBaseline(Scope* scope) override {
    auto* out = scope->NewTensor(output_);
    CHECK(out);
    int64_t nchw[] = {dims_[0], dims_[1], dims_[2], dims_[3]};
    std::vector<int64_t> output_shape(nchw, nchw + 4);
    output_shape.erase(output_shape.begin() + axis_);
    DDim output_dims(output_shape);
    out->Resize(output_dims);
    auto* output_data = out->mutable_data<float>();

    auto* x = scope->FindTensor(input_);
    const auto* x_data = x->data<float>();

    // int in_channel = x_dims
    const int size = dims_[axis_];
    const int in_channel = dims_.count(axis_, dims_.size());
    const int out_channel = output_dims.count(axis_, output_dims.size());
    const int in_stride = dims_.count(axis_ + 1, dims_.size());
    const int out_stride = dims_.count(0, axis_);

    for (int n = 0; n < out_stride; n++) {
      for (int k = 0; k < in_stride; k++) {
        const float* in_ptr = x_data + n * in_channel + k;
        std::vector<std::pair<float, int>> vec;
        vec.resize(size);
        for (int i = 0; i < size; i++) {
          vec[i] = std::make_pair(in_ptr[i * in_stride], i);
        }
        // sort
        std::partial_sort(vec.begin(),
                          vec.begin() + 1,
                          vec.end(),
                          std::greater<std::pair<float, int>>());

        // out
        float* out_ptr = output_data + n * out_channel + k;
        *out_ptr = vec[0].second;
      }
    }
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType("argmax");
    op_desc->SetInput("X", {input_});
    op_desc->SetOutput("Out", {output_});
    op_desc->SetAttr("Axis", axis_);
  }

  void PrepareData() override {
    std::vector<float> data(dims_.production());

    for (int i = 0; i < dims_.production(); i++) {
      float sign = i % 3 == 0 ? -1.0f : 1.0f;
      data[i] = sign * static_cast<float>(i % 128) * 0.013f + 0.001;
    }

    SetCommonTensor(input_, dims_, data.data());
  }
};

TEST(Argmax, precision) {
  // #ifdef LITE_WITH_X86
  //  Place place(TARGET(kX86));
  // #endif
  LOG(INFO) << "test argmax op";
#ifdef LITE_WITH_ARM
  LOG(INFO) << "test argmax arm";
  Place place(TARGET(kARM));

  for (int axis : {0, 1, 2, 3}) {
    for (int n : {1, 3}) {
      for (int c : {3, 6}) {
        for (int h : {9, 18}) {
          for (int w : {9, 18}) {
            std::unique_ptr<arena::TestCase> tester(
                new ArgmaxComputeTester(place, "def", axis, n, c, h, w));
            arena::Arena arena(std::move(tester), place, 2e-5);
            arena.TestPrecision();
          }
        }
      }
    }
  }
#endif
}

}  // namespace lite
}  // namespace paddle
