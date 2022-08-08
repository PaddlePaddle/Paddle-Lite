// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

template <class T>
class SplitTester : public arena::TestCase {
 protected:
  std::string x_ = "X";
  std::vector<std::string> outs_;
  std::string axis_tensor_;
  std::vector<std::string> sections_tensor_list_;
  DDim x_dims_;
#if defined(NNADAPTER_WITH_NVIDIA_TENSORRT)
  int axis_ = 1;
#else
  int axis_ = 0;
#endif
  int num_ = 1;
  std::vector<int> sections_;

 public:
  SplitTester(const Place& place,
              const std::string& alias,
              const DDim& x_dims,
              const int axis = 0,
              const int num = 1,
              const std::vector<int> sections = {},
              const bool use_axis_tensor = false,
              const bool use_sections_tensor_list = false)
      : TestCase(place, alias),
        x_dims_(x_dims),
        axis_(axis),
        num_(num),
        sections_(sections) {
    if (use_axis_tensor) {
      axis_tensor_ = "axis";
    }
    if (use_sections_tensor_list) {
      for (size_t i = 0; i < sections.size(); i++) {
        sections_tensor_list_.push_back("sections" + std::to_string(i));
      }
    }

    int n = num > 0 ? num : static_cast<int>(sections.size());
    for (int i = 0; i < n; i++) {
      outs_.push_back("out" + std::to_string(i));
    }
  }

  void RunBaseline(Scope* scope) override {
    std::vector<Tensor*> outs_tensor;
    for (auto out_name : outs_) {
      auto* out = scope->NewTensor(out_name);
      outs_tensor.push_back(out);
    }
    auto* x = scope->FindTensor(x_);

    const int axis =
        axis_ >= 0 ? axis_ : axis_ + static_cast<int>(x_dims_.size());
    const int num = num_;
    std::vector<int> sections(sections_);
    if (num > 0) {
      sections = std::vector<int>(num, static_cast<int>(x_dims_[axis]) / num);
    } else {
      int len = static_cast<int>(x_dims_[axis]);
      for (auto i : sections) {
        if (i > 0) {
          len -= i;
        }
      }
      if (len > 0) {
        for (size_t i = 0; i < sections.size(); i++) {
          if (sections[i] < 0) {
            sections[i] = len;
            break;
          }
        }
      }
    }

    auto* x_data = x->template data<T>();
    std::vector<int64_t> out_shape = x_dims_.Vectorize();
    int stride = 0;
    for (size_t i = 0; i < sections.size(); i++) {
      out_shape[axis] = sections[i];
      auto* out = outs_tensor[i];
      out->Resize(out_shape);
      auto* out_data = out->template mutable_data<T>();

      int n = 1;
      for (int i = 0; i < axis; i++) {
        n *= static_cast<int>(x_dims_[i]);
      }
      int step_x = x_dims_[axis];
      int step_out = out_shape[axis];
      for (int i = static_cast<int>(out_shape.size()) - 1; i > axis; i--) {
        step_x *= static_cast<int>(out_shape[i]);
        step_out *= static_cast<int>(out_shape[i]);
      }

      auto* in_ptr = x_data + stride;
      for (int i = 0; i < n; i++) {
        memcpy(out_data, in_ptr, sizeof(T) * step_out);
        in_ptr += step_x;
        out_data += step_out;
      }

      stride += step_out;
    }
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType("split");
    op_desc->SetInput("X", {x_});
    if (!axis_tensor_.empty()) {
      op_desc->SetInput("AxisTensor", {axis_tensor_});
    }
    if (!sections_tensor_list_.empty()) {
      op_desc->SetInput("SectionsTensorList", sections_tensor_list_);
    }
    op_desc->SetOutput("Out", outs_);
    op_desc->SetAttr("axis", axis_);
    op_desc->SetAttr("num", num_);
    op_desc->SetAttr("sections", sections_);
  }

  void PrepareData() override {
    std::vector<T> x_data(x_dims_.production());
    fill_data_rand(x_data.data(),
                   static_cast<T>(-10),
                   static_cast<T>(10),
                   x_dims_.production());
    SetCommonTensor(x_, x_dims_, x_data.data());

    if (!axis_tensor_.empty()) {
      std::vector<int> axis_data{axis_};
      SetCommonTensor(axis_tensor_, DDim{{1}}, axis_data.data(), {}, true);
    }

    if (!sections_tensor_list_.empty()) {
      for (size_t i = 0; i < sections_.size(); i++) {
        std::vector<int> sections_data{sections_[i]};
        SetCommonTensor(sections_tensor_list_[i],
                        DDim{{1}},
                        sections_data.data(),
                        {},
                        true);
      }
    }
  }
};

template <class T>
void TestSplit(Place place,
               float abs_error,
               const std::string& alias,
               const DDim& x_dims,
               const int axis = 0,
               const int num = 2,
               const std::vector<int>& sections = {},
               const bool use_axis_tensor = false,
               const bool use_sections_tensor_list = false) {
  std::unique_ptr<arena::TestCase> tester(
      new SplitTester<T>(place,
                         alias,
                         x_dims,
                         axis,
                         num,
                         sections,
                         use_axis_tensor,
                         use_sections_tensor_list));
  arena::Arena arena(std::move(tester), place, abs_error);
  arena.TestPrecision();
}

template <class T = float>
void TestSplitBase(Place place,
                   float abs_error,
                   const std::string& alias = "def") {
  TestSplit<T>(place, abs_error, alias, DDim{{2, 3, 4, 6}}, -1);
}

template <class T = float>
void TestSplitAxis(Place place,
                   float abs_error,
                   const std::string& alias = "def") {
  std::vector<std::vector<int64_t>> x_shapes{{4}, {4, 6, 8, 10}};
  for (auto x_shape : x_shapes) {
    for (auto axis : {-4, -1, 0, 1, 2, 3}) {
      if (axis >= static_cast<int>(x_shape.size()) ||
          axis < -static_cast<int>(x_shape.size())) {
        continue;
      }
#if defined(NNADAPTER_WITH_NVIDIA_TENSORRT)
      if (x_shape.size() == 1 || axis == 0 || axis + x_shape.size() == 0) {
        continue;
      }
#endif
      TestSplit<T>(place, abs_error, alias, DDim(x_shape), axis);
    }
  }
}

template <class T = float>
void TestSplitNum(Place place,
                  float abs_error,
                  const std::string& alias = "def") {
  for (int num : {3, 2}) {
    TestSplit<T>(place, abs_error, alias, DDim{{2, 3, 4, 6}}, -1, num);
  }
}

template <class T = float>
void TestSplitSections(Place place,
                       float abs_error,
                       const std::string& alias = "def") {
  TestSplit<T>(place, abs_error, alias, DDim{{2, 3, 4, 6}}, -1, 0, {1, 5});
  TestSplit<T>(place, abs_error, alias, DDim{{2, 3, 4, 6}}, -1, 0, {2, 3, 1});
}

template <class T = float>
void TestSplitAxisTensor(Place place,
                         float abs_error,
                         const std::string& alias = "def") {
  TestSplit<T>(place, abs_error, alias, DDim{{2, 3, 4, 6}}, -1, 3, {}, true);
}

template <class T = float>
void TestSplitSectionsTensorList(Place place,
                                 float abs_error,
                                 const std::string& alias = "def") {
  TestSplit<T>(place,
               abs_error,
               alias,
               DDim{{2, 3, 4, 6}},
               -1,
               0,
               {3, 2, 1},
               false,
               true);
}

TEST(Split_test, precision) {
  float abs_error;
  Place place;
#if defined(LITE_WITH_NNADAPTER)
  place = TARGET(kNNAdapter);
  abs_error = 2e-5;
#if defined(NNADAPTER_WITH_HUAWEI_ASCEND_NPU)
  abs_error = 1e-2;
  TestSplitBase<float>(place, abs_error);
  TestSplitAxis(place, abs_error);
  TestSplitNum(place, abs_error);
  TestSplitSections(place, abs_error);
  TestSplitAxisTensor(place, abs_error);
  TestSplitSectionsTensorList(place, abs_error);
#elif defined(NNADAPTER_WITH_VERISILICON_TIMVX)
  abs_error = 1e-2;
  TestSplitBase<float>(place, abs_error);
  TestSplitAxis(place, abs_error);
  return;
#elif defined(NNADAPTER_WITH_NVIDIA_TENSORRT)
  abs_error = 1e-2;
  TestSplitBase<float>(place, abs_error);
  TestSplitAxis(place, abs_error);
  TestSplitNum(place, abs_error);
  TestSplitSections(place, abs_error);
  TestSplitAxisTensor(place, abs_error);
  TestSplitSectionsTensorList(place, abs_error);
#elif defined(NNADAPTER_WITH_INTEL_OPENVINO)
  abs_error = 1e-5;
  TestSplitBase<float>(place, abs_error);
  TestSplitAxis(place, abs_error);
  TestSplitNum(place, abs_error);
  TestSplitSections(place, abs_error);
  TestSplitAxisTensor(place, abs_error);
  TestSplitSectionsTensorList(place, abs_error);
#elif defined(NNADAPTER_WITH_CAMBRICON_MLU)
  abs_error = 1e-5;
  TestSplitBase<float>(place, abs_error);
  TestSplitAxis(place, abs_error);
  TestSplitNum(place, abs_error);
  TestSplitSections(place, abs_error);
  TestSplitAxisTensor(place, abs_error);
  TestSplitSectionsTensorList(place, abs_error);
#elif defined(NNADAPTER_WITH_QUALCOMM_QNN)
  abs_error = 1e-2;
  TestSplitBase<float>(place, abs_error);
  TestSplitAxis(place, abs_error);
  TestSplitNum(place, abs_error);
  TestSplitSections(place, abs_error);
  TestSplitAxisTensor(place, abs_error);
  TestSplitSectionsTensorList(place, abs_error);
#else
  return;
#endif
#elif defined(LITE_WITH_X86) || defined(LITE_WITH_ARM)
  place = TARGET(kHost);
  abs_error = 1e-5;
  TestSplitBase<float>(place, abs_error, "def");
  TestSplitBase<int>(place, abs_error, "int32");
  TestSplitBase<int64_t>(place, abs_error, "int64");
  TestSplitAxis(place, abs_error);
  TestSplitNum(place, abs_error);
  TestSplitSections(place, abs_error);
  TestSplitAxisTensor(place, abs_error);
  TestSplitSectionsTensorList(place, abs_error);
#else
  return;
#endif
}

}  // namespace lite
}  // namespace paddle
