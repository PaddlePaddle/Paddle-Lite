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

class ReshapeComputeTester : public arena::TestCase {
 protected:
  // common attributes for this op.
  std::string op_type_ = "reshape2";
  std::string input_ = "x";
  std::string output_ = "out";
  std::string xshape_ = "xshape";
  std::vector<std::string> shape_tensor_vct_;
  std::string shape_tensor_;
  DDim dims_;
  std::vector<int> shape_;
  bool inplace_ = false;

 public:
  ReshapeComputeTester(const Place& place,
                       const std::string& alias,
                       DDim dims,
                       std::vector<int> shape,
                       bool is_shape_tensor_vct = false,
                       bool is_shape_tensor = false,
                       bool is_shape = true)
      : TestCase(place, alias), dims_(dims) {
    if (is_shape_tensor_vct) {
      for (size_t i = 0; i < shape.size(); i++) {
        shape_tensor_vct_.emplace_back(op_type_ + "/shape" +
                                       paddle::lite::to_string(i));
      }
    } else if (is_shape_tensor) {
      shape_tensor_ = op_type_ + "/shape";
    } else if (is_shape) {
      shape_ = shape;
    } else {
      LOG(FATAL) << "must set new shape!";
    }
  }

  void RunBaseline(Scope* scope) override {
    auto* out = scope->NewTensor(output_);
    CHECK(out);

    auto* x = scope->FindTensor(input_);

    std::vector<int> out_shape;
    if (shape_tensor_vct_.size() > 0) {
      for (auto shape_tensor : shape_tensor_vct_) {
        out_shape.push_back(scope->FindTensor(shape_tensor)->data<int>()[0]);
      }
    } else if (!shape_tensor_.empty()) {
      auto shape_tensor = scope->FindTensor(shape_tensor_);
      auto shape_tensor_data = shape_tensor->data<int>();
      out_shape = std::vector<int>(shape_tensor_data,
                                   shape_tensor_data + shape_tensor->numel());
    } else if (!shape_.empty()) {
      out_shape = shape_;
    } else {
      LOG(FATAL) << "must set new shape!";
    }

    std::vector<int64_t> final_out_shape(out_shape.size(), 1);
    int unk_dim_idx = -1;
    int cap = 1;
    for (size_t i = 0; i < out_shape.size(); i++) {
      if (out_shape[i] == -1) {
        CHECK_EQ(unk_dim_idx, -1);
        unk_dim_idx = i;
      } else if (out_shape[i] == 0) {
        CHECK_LE(i, dims_.size());
        final_out_shape[i] = dims_[i];
      } else if (out_shape[i] > 0) {
        final_out_shape[i] = out_shape[i];
      } else {
        LOG(FATAL) << "invalid shape";
      }
      cap *= final_out_shape[i];
    }

    if (unk_dim_idx > -1) {
      final_out_shape[unk_dim_idx] = dims_.production() / cap;
    }

    out->Resize(final_out_shape);

    auto x_data = x->data<float>();
    auto out_data = out->mutable_data<float>();
    memcpy(out_data, x_data, sizeof(float) * dims_.production());

    if (op_type_ == "reshape2") {
      auto* xshape = scope->NewTensor(xshape_);
      auto xshape_dims = dims_.Vectorize();
      xshape_dims.insert(xshape_dims.begin(), 0);
      xshape->Resize(xshape_dims);
    }
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType(op_type_);
    op_desc->SetInput("X", {input_});
    if (shape_tensor_vct_.size() > 0) {
      op_desc->SetInput("ShapeTensor", shape_tensor_vct_);
    } else if (!shape_tensor_.empty()) {
      op_desc->SetInput("Shape", {shape_tensor_});
    } else if (shape_.size() > 0) {
      op_desc->SetAttr("shape", shape_);
    } else {
      LOG(FATAL) << "invalid shape";
    }
    op_desc->SetOutput("Out", {output_});
    if (op_type_ == "reshape2") {
      op_desc->SetOutput("XShape", {xshape_});
    }
    op_desc->SetAttr("inplace", inplace_);
  }

  void PrepareData() override {
    std::vector<float> din(dims_.production());
    fill_data_rand(din.data(), -1.f, 1.f, dims_.production());
    SetCommonTensor(input_, dims_, din.data());

    if (shape_tensor_vct_.size() > 0) {
      for (size_t i = 0; i < shape_.size(); i++) {
        std::vector<int> shape_data{shape_[i]};
        SetCommonTensor(shape_tensor_vct_[i],
                        DDim(std::vector<int64_t>{1}),
                        shape_data.data());
      }
    }
    if (!shape_tensor_.empty()) {
      SetCommonTensor(
          shape_tensor_,
          DDim(std::vector<int64_t>{static_cast<int64_t>(shape_.size())}),
          shape_.data());
    }
  }
};

void TestReshape4D(Place place, float abs_error) {
  DDim dims{{2, 3, 4, 5}};
  std::vector<std::vector<int>> shapes{{5, 4, 3, 2},
                                       {2, 3, 20},
                                       {2, 60},
                                       {120},
                                       {2, 3, -1},
                                       {0, 0, 20},
                                       {0, 0, -1}};
  for (auto shape : shapes) {
    std::unique_ptr<arena::TestCase> tester(
        new ReshapeComputeTester(place, "def", dims, shape));
    arena::Arena arena(std::move(tester), place, abs_error);
    arena.TestPrecision({"xshape"});
  }
}

void TestReshape3D(Place place, float abs_error) {
  DDim dims{{2, 3, 20}};
  std::vector<std::vector<int>> shapes{
      {5, 4, 3, 2}, {2, 3, 20}, {2, 60}, {120}, {2, 3, -1}, {0, 60}, {0, -1}};
  for (auto shape : shapes) {
    std::unique_ptr<arena::TestCase> tester(
        new ReshapeComputeTester(place, "def", dims, shape));
    arena::Arena arena(std::move(tester), place, abs_error);
    arena.TestPrecision({"xshape"});
  }
}

void TestReshape2D(Place place, float abs_error) {
  DDim dims{{6, 20}};
  std::vector<std::vector<int>> shapes{
      {5, 4, 3, 2}, {2, 3, 20}, {2, 60}, {120}, {-1}};
  for (auto shape : shapes) {
    std::unique_ptr<arena::TestCase> tester(
        new ReshapeComputeTester(place, "def", dims, shape));
    arena::Arena arena(std::move(tester), place, abs_error);
    arena.TestPrecision({"xshape"});
  }
}

TEST(Reshape, precision) {
  LOG(INFO) << "test Reshape op";
  float abs_error = 2e-5;
  Place place;
#if defined(LITE_WITH_NPU)
  place = TARGET(kNPU);
  abs_error = 1e-2;  // Using fp16 in NPU
#elif defined(LITE_WITH_ARM)
  place = TARGET(kHost);
#elif defined(LITE_WITH_XPU) && defined(LITE_WITH_XTCL)
  place = TARGET(kXPU);
#elif defined(LITE_WITH_HUAWEI_ASCEND_NPU)
  place = TARGET(kHuaweiAscendNPU);
  abs_error = 1e-2;  // precision_mode default is force_fp16
#else
  return;
#endif

  TestReshape4D(place, abs_error);
  TestReshape3D(place, abs_error);
  TestReshape2D(place, abs_error);
}

}  // namespace lite
}  // namespace paddle
