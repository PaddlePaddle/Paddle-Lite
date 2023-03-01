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

class ArgmaxComputeTester : public arena::TestCase {
 protected:
  // common attributes for this op.
  std::string input_ = "x";
  std::string output_ = "out";
  std::string alias_ = "fp32";
  int64_t axis_ = 0.;
  bool keepdims_ = false;
  int dtype_ = -1;
  DDim dims_{{2, 5, 20, 30}};

 public:
  ArgmaxComputeTester(const Place& place,
                      const std::string& alias,
                      int axis,
                      bool keepdims,
                      int dtype,
                      DDim x_dims)
      : TestCase(place, alias),
        alias_(alias),
        axis_(axis),
        keepdims_(keepdims),
        dtype_(dtype),
        dims_(x_dims) {}

  // template function for RunBaseline according to input_tensor precision type
  // and output tensor precision type(dtype)
  template <typename indtype, typename outdtype>
  void RunBaselineKernel(Scope* scope) {
    auto* out = scope->NewTensor(output_);
    CHECK(out);
    int64_t nchw[] = {dims_[0], dims_[1], dims_[2], dims_[3]};
    std::vector<int64_t> output_shape(nchw, nchw + 4);
    if (keepdims_ == false) {
      output_shape.erase(output_shape.begin() + axis_);
    } else {
      output_shape[axis_] = 1;
    }
    DDim output_dims(output_shape);
    out->Resize(output_dims);
    auto* output_data = out->mutable_data<outdtype>();

    auto* x = scope->FindTensor(input_);
    const auto* x_data = x->data<indtype>();
    // int in_channel = x_dims
    const int size = dims_[axis_];
    const int in_channel = dims_.count(axis_, dims_.size());
    const int out_channel = output_dims.count(axis_, output_dims.size());
    const int in_stride = dims_.count(axis_ + 1, dims_.size());
    const int out_stride = dims_.count(0, axis_);
    for (int n = 0; n < out_stride; n++) {
      for (int k = 0; k < in_stride; k++) {
        const indtype* in_ptr = x_data + n * in_channel + k;
        std::vector<std::pair<indtype, outdtype>> vec;
        vec.resize(size);
        for (int i = 0; i < size; i++) {
          vec[i] = std::make_pair(in_ptr[i * in_stride], i);
        }
        // sort
        std::partial_sort(vec.begin(),
                          vec.begin() + 1,
                          vec.end(),
                          std::greater<std::pair<indtype, outdtype>>());

        // out
        auto* out_ptr = output_data + n * out_channel + k;
        *out_ptr = vec[0].second;
      }
    }
  }

  // template function for RunBaseline according to output tensor precision type
  // (dtype).
  template <typename T>
  void RunBaselineDtypeKernel(Scope* scope) {
    if (alias_ == "fp32" || alias_ == "def") {
      RunBaselineKernel<float, T>(scope);
    } else if (alias_ == "int64") {
      RunBaselineKernel<int64_t, T>(scope);
    } else if (alias_ == "int32") {
      RunBaselineKernel<int32_t, T>(scope);
    } else if (alias_ == "int16") {
      RunBaselineKernel<int16_t, T>(scope);
    } else if (alias_ == "uint8") {
      RunBaselineKernel<uint8_t, T>(scope);
    } else {
      LOG(ERROR) << "Not support alias: " << alias_;
    }
  }

  void RunBaseline(Scope* scope) override {
    if (axis_ < 0) {
      axis_ += static_cast<int>(dims_.size());
    }
    if (dtype_ == -1 || dtype_ == 3) {
      RunBaselineDtypeKernel<int64_t>(scope);
    } else if (dtype_ == 2) {
      RunBaselineDtypeKernel<int32_t>(scope);
    } else {
      LOG(FATAL) << "Error: unsupported dtype:" << dtype_;
    }
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) override {
    op_desc->SetType("arg_max");
    op_desc->SetInput("X", {input_});
    op_desc->SetOutput("Out", {output_});
    op_desc->SetAttr("axis", axis_);
    op_desc->SetAttr("keepdims", keepdims_);
    op_desc->SetAttr("dtype", dtype_);
  }

  void PrepareData() override {
    if (alias_ == "fp32" || alias_ == "def") {
      std::vector<float> data(dims_.production());
      for (int i = 0; i < dims_.production(); i++) {
        float sign = i % 3 == 0 ? -1.0f : 1.0f;
        data[i] = sign * static_cast<float>(i % 128) * 0.013f + 0.001;
      }
      SetCommonTensor(input_, dims_, data.data());
    } else if (alias_ == "int64") {
      std::vector<int64_t> data(dims_.production());
      for (int i = 0; i < dims_.production(); i++) {
        float sign = i % 3 == 0 ? -1.0f : 1.0f;
        data[i] = sign * static_cast<int64_t>(i % 128);
      }
      SetCommonTensor(input_, dims_, data.data());
    } else if (alias_ == "int32") {
      std::vector<int32_t> data(dims_.production());
      for (int i = 0; i < dims_.production(); i++) {
        float sign = i % 3 == 0 ? -1.0f : 1.0f;
        data[i] = sign * static_cast<int32_t>(i % 64);
      }
      SetCommonTensor(input_, dims_, data.data());
    } else if (alias_ == "int16") {
      std::vector<int16_t> data(dims_.production());
      for (int i = 0; i < dims_.production(); i++) {
        float sign = i % 3 == 0 ? -1.0f : 1.0f;
        data[i] = sign * static_cast<int16_t>(i % 32);
      }
      SetCommonTensor(input_, dims_, data.data());
    } else if (alias_ == "uint8") {
      std::vector<uint8_t> data(dims_.production());
      for (int i = 0; i < dims_.production(); i++) {
        data[i] = static_cast<uint8_t>(i % 32);
      }
      SetCommonTensor(input_, dims_, data.data());
    }
  }
};

void TestArgmax(const Place& place,
                std::vector<std::string> aliases,
                std::vector<int> dtype_list) {
  for (int axis : {-1, -2, 0, 2}) {
    for (bool keepdims : {false, true}) {
      for (int dtype : dtype_list) {
        for (auto x_shape :
             std::vector<std::vector<int64_t>>{{1, 2, 3, 4}, {2, 3, 4, 5}}) {
          int x_size = x_shape.size();
          if (axis < -x_size || axis >= x_size) continue;
#ifdef NNADAPTER_WITH_NVIDIA_TENSORRT
          if (axis == 0 || axis == -static_cast<int>(x_shape.size())) continue;
#endif
          for (std::string alias : aliases) {
            std::unique_ptr<arena::TestCase> tester(new ArgmaxComputeTester(
                place, alias, axis, keepdims, dtype, DDim(x_shape)));
            arena::Arena arena(std::move(tester), place, 2e-5);
            arena.TestPrecision();
          }
        }
      }
    }
  }
}

TEST(Argmax, precision) {
  Place place;
  std::vector<std::string> aliases{"fp32", "int64", "int32", "int16", "uint8"};
#if defined(LITE_WITH_NNADAPTER)
  place = TARGET(kNNAdapter);
  aliases = {"def"};
#if defined(NNADAPTER_WITH_HUAWEI_ASCEND_NPU)
#elif defined(NNADAPTER_WITH_HUAWEI_KIRIN_NPU)
  TestArgmax(place, aliases, {2});
  return;
#elif defined(NNADAPTER_WITH_NVIDIA_TENSORRT)
  TestArgmax(place, aliases, {2});
  return;
#elif defined(NNADAPTER_WITH_INTEL_OPENVINO)
  TestArgmax(place, aliases, {2});
  return;
#elif defined(NNADAPTER_WITH_CAMBRICON_MLU)
  TestArgmax(place, aliases, {2});
  return;
#elif defined(NNADAPTER_WITH_QUALCOMM_QNN)
  TestArgmax(place, aliases, {2});
  return;
#elif defined(NNADAPTER_WITH_VERISILICON_TIMVX)
  TestArgmax(place, aliases, {2});
  return;
#else
  return;
#endif
#elif defined(LITE_WITH_ARM)
  place = TARGET(kARM);
#elif defined(LITE_WITH_X86)
  place = TARGET(kHost);
#else
  return;
#endif

  TestArgmax(place, aliases, {-1, 2, 3});
}

}  // namespace lite
}  // namespace paddle
