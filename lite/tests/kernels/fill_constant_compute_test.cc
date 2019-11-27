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

class FillConstantComputeTester : public arena::TestCase {
 protected:
  // common attributes for this op.
  std::string out_ = "out";
  int dtype_{static_cast<int>(VarDescAPI::VarDataType::FP32)};
  std::vector<int64_t> shape_{};
  std::string shape_tensor_ = "ShapeTensor";
  std::vector<std::string> shape_tensor_list_;
  bool is_use_shape_tensor_{false};
  bool is_use_shape_tensor_list_{false};

  float value_{0.0f};
  // useless for x86, keep it for compatibility
  bool force_cpu_{false};
  // DDim shape_tensor_data{{5, 3}};
  std::vector<int32_t> shape_tensor_data;
  DDim shape_test{{1, 2}};

 public:
  FillConstantComputeTester(const Place& place,
                            const std::string& alias,
                            std::vector<int64_t> shape,
                            const bool is_use_shape_tensor,
                            const bool is_use_shape_tensor_list,
                            float value,
                            bool force_cpu)
      : TestCase(place, alias) {
    shape_ = shape;
    value_ = value;
    force_cpu_ = force_cpu;
    is_use_shape_tensor_ = is_use_shape_tensor;
    is_use_shape_tensor_list_ = is_use_shape_tensor_list;

    for (int i = 0; i < shape_test.size(); i++) {
      shape_tensor_data.push_back(i + 1);
    }
  }

  void RunBaseline(Scope* scope) override {
    auto* out = scope->NewTensor(out_);
    DDim output_dims{shape_};
    if (is_use_shape_tensor_) {
      auto* temp_shape = scope->FindTensor(shape_tensor_);
      auto* shape_data = temp_shape->data<int>();
      auto vec_shape =
          std::vector<int64_t>(shape_data, shape_data + temp_shape->numel());
      output_dims.ConstructFrom(vec_shape);
    }
    if (is_use_shape_tensor_list_) {
      std::vector<int64_t> vec_shape;
      for (int i = 0; i < shape_tensor_list_.size(); i++) {
        auto* temp_shape = scope->FindTensor(shape_tensor_list_[i]);
        vec_shape.push_back(*temp_shape->data<int>());
      }

      output_dims.ConstructFrom(vec_shape);
    }
    out->Resize(output_dims);

    auto* output_data = out->mutable_data<float>();
    for (int i = 0; i < out->numel(); i++) {
      output_data[i] = value_;
    }
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    LOG(INFO) << "PrepareOpDesc";

    op_desc->SetType("fill_constant");
    op_desc->SetAttr("dtype", dtype_);
    op_desc->SetAttr("shape", shape_);
    op_desc->SetAttr("value", value_);
    op_desc->SetAttr("force_cpu", force_cpu_);
    if (is_use_shape_tensor_) {
      op_desc->SetInput("ShapeTensor", {shape_tensor_});
    }
    if (is_use_shape_tensor_list_) {
      // std::vector<std::string> shape_tensor_list_;
      for (int i = 0; i < shape_test.size(); ++i) {
        shape_tensor_list_.push_back("shape_tensor_list_" + std::to_string(i));
      }
      op_desc->SetInput("ShapeTensorList", {shape_tensor_list_});
    }
    op_desc->SetOutput("Out", {out_});
  }

  void PrepareData() override {
    if (is_use_shape_tensor_) {
      // std::vector<int64_t> temp = x_dims_.data();
      // int64_t* data = temp.data();
      SetCommonTensor(shape_tensor_, shape_test, shape_tensor_data.data());
    }
    if (is_use_shape_tensor_list_) {
      Scope& scope_ = this->scope();
      for (int i = 0; i < shape_test.size(); ++i) {
        auto* tensor =
            scope_.NewTensor("shape_tensor_list_" + std::to_string(i));
        tensor->Resize(DDim({1}));
        auto* d = tensor->mutable_data<int>();
        d[0] = shape_tensor_data[i];
      }
    }
  }
};

TEST(fill_constant, precision) {
  LOG(INFO) << "test fill_constant op, kARM";
#ifdef LITE_WITH_ARM
  Place place(TARGET(kARM));
  std::vector<int64_t> shape{1, 2};

  for (int dtype : {static_cast<int>(VarDescAPI::VarDataType::INT32)}) {
    for (float value : {1, 2}) {
      for (bool is_use_shape_tensor_list : {false, true}) {
        for (bool is_use_shape_tensor : {false, true}) {
          if (is_use_shape_tensor && is_use_shape_tensor_list) break;
          LOG(INFO) << "value:" << value
                    << ", is_use_shape_tensor:" << is_use_shape_tensor
                    << ", is_use_shape_tensor_list:"
                    << is_use_shape_tensor_list;

          std::unique_ptr<arena::TestCase> tester(
              new FillConstantComputeTester(place,
                                            "def",
                                            shape,
                                            is_use_shape_tensor,
                                            is_use_shape_tensor_list,
                                            value,
                                            false));
          arena::Arena arena(std::move(tester), place, 2e-5);
          arena.TestPrecision();
        }
      }
    }
  }
#endif

#ifdef LITE_WITH_X86
  Place place(TARGET(kX86));
  LOG(INFO) << "test concate op, x86";
  for (int axis : {1, 2}) {
    for (bool is_use_axis_tensor : {false, true}) {
      LOG(INFO) << "axis:" << axis
                << ", is_use_axis_tensor:" << is_use_axis_tensor;
      std::unique_ptr<arena::TestCase> tester(
          new ConcateComputeTester(place, "def", axis, is_use_axis_tensor));
      arena::Arena arena(std::move(tester), place, 2e-5);
      arena.TestPrecision();
    }
  }

#endif
}

}  // namespace lite
}  // namespace paddle
