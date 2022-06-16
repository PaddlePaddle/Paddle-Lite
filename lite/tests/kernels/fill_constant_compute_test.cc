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

namespace paddle {
namespace lite {

class FillConstantComputeTester : public arena::TestCase {
 protected:
  // common attributes for this op.
  std::string out_ = "out";
  std::string shape_tensor_ = "shape_tensor";
  std::string value_tensor_ = "value_tensor";
  std::vector<std::string> shape_tensor_list_{};

  std::vector<int64_t> shape_{};
  std::vector<float> tensor_value_{};
  float value_{0.0f};
  int dtype_{static_cast<int>(VarDescAPI::VarDataType::FP32)};

  bool is_use_shape_tensor_{false};
  bool is_use_shape_tensor_list_{false};
  // useless for x86, keep it for compatibility
  bool force_cpu_{false};

 public:
  FillConstantComputeTester(const Place& place,
                            const std::string& alias,
                            std::vector<int64_t> shape,
                            std::vector<float> tensor_value,
                            float value,
                            int dtype,
                            const bool is_use_shape_tensor = false,
                            const bool is_use_shape_tensor_list = false)
      : TestCase(place, alias),
        shape_(shape),
        tensor_value_(tensor_value),
        value_(value),
        dtype_(dtype),
        is_use_shape_tensor_(is_use_shape_tensor),
        is_use_shape_tensor_list_(is_use_shape_tensor_list) {
    if (is_use_shape_tensor_list) {
      for (int i = 0; i < shape.size(); i++) {
        shape_tensor_list_.push_back(shape_tensor_ +
                                     paddle::lite::to_string(i));
      }
    }
  }

  void RunBaseline(Scope* scope) override {
    auto* out = scope->NewTensor(out_);
    std::vector<int64_t> out_shape;
    if (is_use_shape_tensor_) {
      auto* shape_tensor = scope->FindTensor(shape_tensor_);
      auto* shape_tensor_data = shape_tensor->data<int>();
      out_shape = std::vector<int64_t>(
          shape_tensor_data, shape_tensor_data + shape_tensor->numel());
    } else if (is_use_shape_tensor_list_) {
      for (int i = 0; i < shape_tensor_list_.size(); i++) {
        auto* shape_tensor = scope->FindTensor(shape_tensor_list_[i]);
        out_shape.push_back(shape_tensor->data<int>()[0]);
      }
    } else {
      out_shape = shape_;
    }
#ifdef NNADAPTER_WITH_NVIDIA_TENSORRT
    // Trt out should have batchsize
    out_shape.insert(out_shape.begin(), 1);
#endif
    out->Resize(out_shape);

    switch (dtype_) {
      case static_cast<int>(VarDescAPI::VarDataType::BOOL): {
        FillConstData<bool>(scope);
        break;
      }
      case static_cast<int>(VarDescAPI::VarDataType::INT16): {
        FillConstData<int16_t>(scope);
        break;
      }
      case static_cast<int>(VarDescAPI::VarDataType::INT32): {
        FillConstData<int32_t>(scope);
        break;
      }
      case static_cast<int>(VarDescAPI::VarDataType::INT64): {
        FillConstData<int64_t>(scope);
        break;
      }
      case static_cast<int>(VarDescAPI::VarDataType::FP32): {
        FillConstData<float>(scope);
        break;
      }
      default: {
        LOG(FATAL) << "Attribute dtype in fill_constant op test"
                      "must be 0[bool] or 1[int16] or 2[int32] or "
                      "3[int64] or 5[fp32] for baseline: "
                   << dtype_;
        break;
      }
    }
  }

  template <typename T>
  void FillConstData(Scope* scope) {
    if (!tensor_value_.empty()) {
      value_ = tensor_value_[0];
    }
    auto* out = scope->FindMutableTensor(out_);
    auto* output_data = out->mutable_data<T>();
    for (int i = 0; i < out->numel(); i++) {
      output_data[i] = value_;
    }
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType("fill_constant");
    if (!tensor_value_.empty()) {
      op_desc->SetInput("ValueTensor", {value_tensor_});
    }
    if (is_use_shape_tensor_) {
      op_desc->SetInput("ShapeTensor", {shape_tensor_});
    } else if (is_use_shape_tensor_list_) {
      op_desc->SetInput("ShapeTensorList", shape_tensor_list_);
    } else {
      op_desc->SetAttr("shape", shape_);
    }
    op_desc->SetOutput("Out", {out_});
    op_desc->SetAttr("dtype", dtype_);
    op_desc->SetAttr("value", value_);
    op_desc->SetAttr("force_cpu", force_cpu_);
  }

  void PrepareData() override {
    if (!tensor_value_.empty()) {
      std::vector<float> ddata_tensor{tensor_value_.begin(),
                                      tensor_value_.end()};
      SetCommonTensor(value_tensor_,
                      DDim({static_cast<int64_t>(tensor_value_.size())}),
                      ddata_tensor.data());
    }
    if (is_use_shape_tensor_) {
      std::vector<int> dshape_tensor(shape_.begin(), shape_.end());
      SetCommonTensor(shape_tensor_,
                      DDim({static_cast<int64_t>(shape_.size())}),
                      dshape_tensor.data());
    }
    if (is_use_shape_tensor_list_) {
      for (int i = 0; i < shape_.size(); ++i) {
        std::vector<int> dshape_tensor{static_cast<int>(shape_[i])};
        SetCommonTensor(shape_tensor_list_[i], DDim({1}), dshape_tensor.data());
      }
    }
  }
};

void TestFillConstantShape(Place place, float abs_error) {
  std::vector<std::vector<int64_t>> out_shapes{
      {2, 3, 4, 5}, {2, 3, 4}, {3, 4}, {4}};
  for (auto out_shape : out_shapes) {
    std::vector<float> value_tensor = {2.f};
    std::unique_ptr<arena::TestCase> tester(new FillConstantComputeTester(
        place,
        "def",
        out_shape,
        value_tensor,
        1.f,
        static_cast<int>(VarDescAPI::VarDataType::FP32)));
    arena::Arena arena(std::move(tester), place, abs_error);
    arena.TestPrecision();
  }
}

void TestFillConstantValue(Place place, float abs_error) {
  std::vector<float> values{-1., 0., 3.5};
  for (auto value : values) {
    std::unique_ptr<arena::TestCase> tester(new FillConstantComputeTester(
        place,
        "def",
        {2, 3},
        {},
        value,
        static_cast<int>(VarDescAPI::VarDataType::FP32)));
    arena::Arena arena(std::move(tester), place, abs_error);
    arena.TestPrecision();
  }

#if defined(LITE_WITH_XPU)
  std::vector<bool> values_bool{true, false};
  for (auto value : values_bool) {
    std::unique_ptr<arena::TestCase> tester(new FillConstantComputeTester(
        place,
        "def",
        {2, 3},
        {},
        value,
        static_cast<int>(VarDescAPI::VarDataType::BOOL)));
    arena::Arena arena(std::move(tester), place, abs_error);
    arena.TestPrecision();
  }
  std::vector<int32_t> values_int32{1, 0x1fffffff, -123};
  for (auto value : values_int32) {
    std::unique_ptr<arena::TestCase> tester(new FillConstantComputeTester(
        place,
        "def",
        {2, 3},
        {},
        value,
        static_cast<int>(VarDescAPI::VarDataType::INT32)));
    arena::Arena arena(std::move(tester), place, abs_error);
    arena.TestPrecision();
  }
  std::vector<int64_t> values_int64{1, 0x1fffffff, -123};
  for (auto value : values_int64) {
    std::unique_ptr<arena::TestCase> tester(new FillConstantComputeTester(
        place,
        "def",
        {2, 3},
        {},
        value,
        static_cast<int>(VarDescAPI::VarDataType::INT64)));
    arena::Arena arena(std::move(tester), place, abs_error);
    arena.TestPrecision();
  }
#endif
}

void TestFillConstantShapeTensor(Place place, float abs_error) {
  std::unique_ptr<arena::TestCase> tester(new FillConstantComputeTester(
      place,
      "def",
      {2, 3, 4},
      {},
      1.f,
      static_cast<int>(VarDescAPI::VarDataType::FP32),
      true));
  arena::Arena arena(std::move(tester), place, abs_error);
  arena.TestPrecision();
}

void TestFillConstantShapeTensorList(Place place, float abs_error) {
  std::unique_ptr<arena::TestCase> tester(new FillConstantComputeTester(
      place,
      "def",
      {2, 3, 4},
      {},
      1.f,
      static_cast<int>(VarDescAPI::VarDataType::FP32),
      false,
      true));
  arena::Arena arena(std::move(tester), place, abs_error);
  arena.TestPrecision();
}

TEST(fill_constant, precision) {
  Place place;
  float abs_error = 1e-5;
#if defined(LITE_WITH_NNADAPTER)
  place = TARGET(kNNAdapter);
#if defined(NNADAPTER_WITH_HUAWEI_ASCEND_NPU)
  abs_error = 1e-2;
  TestFillConstantShape(place, abs_error);
  return;
#elif defined(NNADAPTER_WITH_NVIDIA_TENSORRT)
  abs_error = 1e-2;
  TestFillConstantValue(place, abs_error);
  return;
#elif defined(NNADAPTER_WITH_INTEL_OPENVINO)
  TestFillConstantShape(place, abs_error);
  TestFillConstantValue(place, abs_error);
  return;
#else
  return;
#endif
#elif defined(LITE_WITH_NPU)
  place = TARGET(kNPU);
  abs_error = 1e-2;  // use fp16 in npu
#elif defined(LITE_WITH_XPU)
  place = TARGET(kXPU);
#elif defined(LITE_WITH_ARM) || defined(LITE_WITH_X86)
  place = TARGET(kHost);
#else
  return;
#endif

  TestFillConstantShape(place, abs_error);
  TestFillConstantValue(place, abs_error);
  TestFillConstantShapeTensor(place, abs_error);
  TestFillConstantShapeTensorList(place, abs_error);
}

}  // namespace lite
}  // namespace paddle
