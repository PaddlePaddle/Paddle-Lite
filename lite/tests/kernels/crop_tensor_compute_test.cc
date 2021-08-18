// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

template <class T>
void slice_ref(const T* input,
               const std::vector<int64_t>& in_dims,
               const std::vector<int>& axes,
               const std::vector<int>& starts,
               const std::vector<int>& ends,
               T* out) {
  auto out_dims = in_dims;
  std::vector<int> real_starts(in_dims.size(), 0);
  std::vector<int> real_ends(in_dims.size(), 0);
  std::vector<int> real_step(in_dims.size(), 0);
  for (size_t i = 0; i < in_dims.size(); i++) {
    real_ends[i] = in_dims[i];
  }
  for (size_t i = 0; i < axes.size(); i++) {
    int dim_value = in_dims[axes[i]];
    if (dim_value > 0) {
      int start = starts[i] < 0 ? (starts[i] + dim_value) : starts[i];
      int end = ends[i] < 0 ? (ends[i] + dim_value) : ends[i];
      start = std::max(start, 0);
      end = std::max(end, 0);
      end = std::min(end, dim_value);
      out_dims[axes[i]] = end - start;
      real_starts[axes[i]] = start;
      real_ends[axes[i]] = end;
    }
  }
  const int LEN = in_dims.size();
  std::vector<int> dst_step(LEN);
  for (size_t i = 0; i < in_dims.size(); ++i) {
    dst_step[i] = 1;
  }
  std::vector<int> src_step(LEN);
  for (size_t i = 0; i < in_dims.size(); ++i) {
    src_step[i] = 1;
  }
  int out_num = out_dims[in_dims.size() - 1];
  for (int i = in_dims.size() - 2; i >= 0; i--) {
    dst_step[i] = out_dims[i + 1] * dst_step[i + 1];
    src_step[i] = in_dims[i + 1] * src_step[i + 1];
    out_num *= out_dims[i];
  }

  for (int dst_id = 0; dst_id < out_num; dst_id++) {
    int src_id = 0;
    int index_id = dst_id;
    for (size_t j = 0; j < out_dims.size(); j++) {
      int cur_id = index_id / dst_step[j];
      index_id = index_id % dst_step[j];
      src_id += (cur_id + real_starts[j]) * src_step[j];
    }
    out[dst_id] = input[src_id];
  }
}

template <class T>
class CropTensorComputeTester : public arena::TestCase {
 protected:
  std::string x_ = "X";
  std::string shape_;
  std::string offsets_;
  std::string shapetensor_;
  std::string offsetstensor_;
  std::string out_ = "Out";
  std::vector<int> attr_shape_;
  std::vector<int> attr_offsets_;
  DDim x_dims_;

 public:
  CropTensorComputeTester(const Place& place,
                          const std::string& alias,
                          const std::vector<int>& shape,
                          const std::vector<int>& offset,
                          const DDim& x_dims,
                          bool has_shape = false,
                          bool has_offsets = false,
                          bool has_shapetensor = false,
                          bool has_offsetstensor = false)
      : TestCase(place, alias),
        attr_shape_(shape),
        attr_offsets_(offset),
        x_dims_(x_dims) {
    if (has_shape) shape_ = "Shape";
    if (has_offsets) offsets_ = "Offsets";
    if (has_shapetensor) shapetensor_ = "ShapeTensor";
    if (has_offsetstensor) offsetstensor_ = "OffsetsTensor";
  }

  void RunBaseline(Scope* scope) override {
    auto* out = scope->NewTensor(out_);
    auto* x = scope->FindTensor(x_);
    std::vector<int64_t> out_shape(attr_shape_.begin(), attr_shape_.end());
    out->Resize(out_shape);
    auto* out_data = out->template mutable_data<T>();
    auto x_shape = x->dims().Vectorize();
    auto* x_data = x->template data<T>();

    std::vector<int> starts = attr_offsets_;
    std::vector<int> ends;
    std::vector<int> axes;
    for (size_t i = 0; i < starts.size(); i++) {
      ends.push_back(starts[i] + attr_shape_[i]);
      axes.push_back(i);
    }

    slice_ref(x_data, x_shape, axes, starts, ends, out_data);
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType("crop_tensor");
    op_desc->SetInput("X", {x_});
    if (!shape_.empty()) op_desc->SetInput("Shape", {shape_});
    if (!offsets_.empty()) op_desc->SetInput("Offsets", {offsets_});
    if (!shapetensor_.empty()) op_desc->SetInput("ShapeTensor", {shapetensor_});
    if (!offsetstensor_.empty())
      op_desc->SetInput("OffsetsTensor", {offsetstensor_});
    op_desc->SetOutput("Out", {out_});
    op_desc->SetAttr("shape", attr_shape_);
    op_desc->SetAttr("offsets", attr_offsets_);
  }

  void PrepareData() override {
    std::vector<T> x_data(x_dims_.production());
    for (int64_t i = 0; i < x_dims_.production(); i++) {
      x_data[i] = i;
    }
    SetCommonTensor(x_, x_dims_, x_data.data());

    if (!shape_.empty()) {
      SetCommonTensor(shape_,
                      DDim({static_cast<int64_t>(attr_shape_.size())}),
                      attr_shape_.data());
    }

    if (!offsets_.empty()) {
      SetCommonTensor(offsets_,
                      DDim({static_cast<int64_t>(attr_offsets_.size())}),
                      attr_offsets_.data());
    }

    if (!shapetensor_.empty()) {
      std::vector<DDim> shapetensor_dims(attr_shape_.size(), DDim{{1}});
      std::vector<std::vector<int>> shapetensor_data;
      for (size_t i = 0; i < attr_shape_.size(); i++) {
        shapetensor_data.push_back({attr_shape_[i]});
      }
      SetCommonTensorList(shapetensor_, shapetensor_dims, shapetensor_data);
    }

    if (!offsetstensor_.empty()) {
      std::vector<DDim> offsetstensor_dims(attr_offsets_.size(), DDim{{1}});
      std::vector<std::vector<int>> offsetstensor_data;
      for (size_t i = 0; i < attr_offsets_.size(); i++) {
        offsetstensor_data.push_back({attr_offsets_[i]});
      }
      SetCommonTensorList(
          offsetstensor_, offsetstensor_dims, offsetstensor_data);
    }
  }
};

template <class T = float>
void TestCropTensor(Place place, float abs_error = 1e-5) {
  place.precision = lite_api::PrecisionTypeTrait<float>::Type();
  std::string alias = "def";
  if (lite_api::PrecisionTypeTrait<T>::Type() == PrecisionType::kInt32) {
    alias = "int32_precision";
  }

  // test 1D
  std::unique_ptr<arena::TestCase> tester_1d(
      new CropTensorComputeTester<T>(place, alias, {1}, {3}, DDim({4})));
  arena::Arena arena_1d(std::move(tester_1d), place, abs_error);
  arena_1d.TestPrecision();

  // test 4D
  std::unique_ptr<arena::TestCase> tester_4d(new CropTensorComputeTester<T>(
      place, alias, {1, 1, 2, 3}, {1, 0, 2, 1}, DDim({2, 3, 4, 5})));
  arena::Arena arena_4d(std::move(tester_4d), place, abs_error);
  arena_4d.TestPrecision();
}

template <class T = float>
void TestCropTensorShape(Place place, float abs_error = 1e-5) {
  place.precision = lite_api::PrecisionTypeTrait<T>::Type();
  std::unique_ptr<arena::TestCase> tester(new CropTensorComputeTester<T>(
      place, "def", {1, 1, 2, 3}, {1, 0, 2, 1}, DDim({2, 3, 4, 5}), true));
  arena::Arena arena(std::move(tester), place, abs_error);
  arena.TestPrecision();
}

template <class T = float>
void TestCropTensorOffsets(Place place, float abs_error = 1e-5) {
  place.precision = lite_api::PrecisionTypeTrait<T>::Type();
  std::unique_ptr<arena::TestCase> tester(
      new CropTensorComputeTester<T>(place,
                                     "def",
                                     {1, 1, 2, 3},
                                     {1, 0, 2, 1},
                                     DDim({2, 3, 4, 5}),
                                     false,
                                     true));
  arena::Arena arena(std::move(tester), place, abs_error);
  arena.TestPrecision();
}

template <class T = float>
void TestCropTensorShapeTensor(Place place, float abs_error = 1e-5) {
  place.precision = lite_api::PrecisionTypeTrait<T>::Type();
  std::unique_ptr<arena::TestCase> tester(
      new CropTensorComputeTester<T>(place,
                                     "def",
                                     {1, 1, 2, 3},
                                     {1, 0, 2, 1},
                                     DDim({2, 3, 4, 5}),
                                     false,
                                     false,
                                     true));
  arena::Arena arena(std::move(tester), place, abs_error);
  arena.TestPrecision();
}

template <class T = float>
void TestCropTensorOffsetsTensor(Place place, float abs_error = 1e-5) {
  place.precision = lite_api::PrecisionTypeTrait<T>::Type();
  std::unique_ptr<arena::TestCase> tester(
      new CropTensorComputeTester<T>(place,
                                     "def",
                                     {1, 1, 2, 3},
                                     {1, 0, 2, 1},
                                     DDim({2, 3, 4, 5}),
                                     false,
                                     false,
                                     false,
                                     true));
  arena::Arena arena(std::move(tester), place, abs_error);
  arena.TestPrecision();
}

TEST(crop_tensor, precision) {
  Place place;
#if defined(LITE_WITH_ARM) || defined(LITE_WITH_X86)
  place = TARGET(kHost);
#else
  return;
#endif

  TestCropTensor<float>(place);
  TestCropTensor<int>(place);
  TestCropTensorShape<float>(place);
  TestCropTensorOffsets<float>(place);
  TestCropTensorShapeTensor<float>(place);
  TestCropTensorOffsetsTensor<float>(place);
}

}  // namespace lite
}  // namespace paddle
