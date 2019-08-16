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

class CropComputeTester : public arena::TestCase {
 protected:
  // common attributes for this op.
  std::string input_ = "X";
  std::string output_ = "Out";

  DDim dims_{{1, 32, 113, 113}};
  std::vector<int> offsets_;
  std::vector<int> shape_;

 public:
  CropComputeTester(const Place& place,
                    const std::string& alias,
                    std::vector<int> offsets,
                    std::vector<int> shape)
      : TestCase(place, alias), offsets_(offsets), shape_(shape) {}

  void RunBaseline(Scope* scope) override {
    LOG(INFO) << "into runbase";
    auto* out = scope->NewTensor(output_);
    LOG(INFO) << "1";
    CHECK(out);
    LOG(INFO) << "2";
    CHECK_EQ(shape_.size(), 4) << "shape size is" << shape_.size();
    lite::DDim output_shape(dims_);
    LOG(INFO) << "2.1";
    output_shape[0] = dims_[0];
    LOG(INFO) << "2.2";
    output_shape[1] = shape_[1];
    LOG(INFO) << "2.3";
    output_shape[2] = shape_[2];
    output_shape[3] = shape_[3];
    LOG(INFO) << "2.4";
    out->Resize(output_shape);
    LOG(INFO) << "3";

    auto* x = scope->FindTensor(input_);
    LOG(INFO) << "into middle";
    CHECK_EQ(shape_.size(), 4) << "shape size is" << shape_.size();
    int c_off = offsets_[1];
    int h_off = offsets_[2];
    int w_off = offsets_[3];
    int c_end = shape_[1] + c_off;
    int h_end = shape_[2] + h_off;
    int w_end = shape_[3] + w_off;

    int num = dims_[0];
    int in_c = dims_[1];
    int in_h = dims_[2];
    int in_w = dims_[3];
    const float* ptr_in = x->data<float>();
    float* ptr_out = out->mutable_data<float>();
    for (int i = 0; i < num; ++i) {
      int offset_n = i * in_c * in_h * in_w;
      for (int j = c_off; j < c_end; ++j) {
        int offset_c = offset_n + j * in_h * in_w;
        for (int k = h_off; k < h_end; ++k) {
          int offset_h = offset_c + k * in_w;
          for (int l = w_off; l < w_end; ++l) {
            ptr_out[0] = ptr_in[offset_h + l];
            ptr_out++;
          }
        }
      }
    }
    LOG(INFO) << "get out of runbase";
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType("crop");
    op_desc->SetInput("X", {input_});
    op_desc->SetOutput("Out", {output_});
    op_desc->SetAttr("offsets", offsets_);
    op_desc->SetAttr("shape", shape_);
  }

  void PrepareData() override {
    std::vector<float> data(dims_.production());

    for (int i = 0; i < dims_.production(); i++) {
      data[i] = i * 1;
    }
    SetCommonTensor(input_, dims_, data.data());
  }
};

void TestCrop(const Place& place) {
  std::vector<int> offset = {0, 0, 1, 1};
  std::vector<int> shape = {-1, 32, 112, 112};
  std::unique_ptr<arena::TestCase> tester(
      new CropComputeTester(place, "def", offset, shape));
  arena::Arena arena(std::move(tester), place, 2e-5);
  arena.TestPrecision();
}

TEST(Crop, precision) {
#ifdef LITE_WITH_X86
  Place place(TARGET(kX86));
#endif
#ifdef LITE_WITH_ARM
  Place place(TARGET(kARM));
  TestCrop(place);
#endif
}

}  // namespace lite
}  // namespace paddle
