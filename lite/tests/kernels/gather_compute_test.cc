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

template <typename IndexType, typename DataType>
void GatherFunc(const operators::GatherParam& param) {
  auto src_dims = param.X->dims();
  auto index_size = param.Index->dims()[0];
  auto* p_src = param.X->data<DataType>();
  const IndexType* p_index = param.Index->data<IndexType>();
  auto* p_output = param.Out->mutable_data<DataType>();

  int slice_size = 1;
  for (size_t i = 1; i < src_dims.size(); ++i) {
    slice_size *= src_dims[i];
  }
  for (int i = 0; i < index_size; ++i) {
    IndexType index_ = p_index[i];
    memcpy(p_output + i * slice_size,
           p_src + index_ * slice_size,
           slice_size * sizeof(DataType));
  }
}

template <typename IndexType, typename AxisType, typename DataType>
void GatherV2Func(const operators::GatherParam& param, const int axis) {
  auto* index_data = param.Index->data<IndexType>();
  auto* input_data = param.X->data<DataType>();
  auto* out_data = param.Out->mutable_data<DataType>();

  int index_size = param.Index->numel();
  int input_size = param.X->numel();
  if (input_size == 0) return;
  auto input_dim = param.X->dims();
  const int axis_index = axis;
  int input_index_dim_size = input_dim[axis_index];
  for (int i = 0; i < index_size; i++) {
    LOG(INFO) << lite::string_format("index_data[%d]: %lld", i, index_data[i]);
    CHECK_LT(index_data[i], input_index_dim_size)
        << "The element of Index must be less than the size of "
        << "dim size of axis dim";
  }

  int inner_dim_size = 1;
  int outer_dim_size = 1;
  for (int i = 0; i < axis_index; i++) {
    inner_dim_size *= input_dim[i];
  }
  for (size_t i = axis_index + 1; i < input_dim.size(); i++) {
    outer_dim_size *= input_dim[i];
  }

  int out_index = 0;
  for (int i = 0; i < inner_dim_size; i++) {
    for (int j = 0; j < index_size; j++) {
      for (int k = 0; k < outer_dim_size; k++) {
        int index = k + index_data[j] * outer_dim_size +
                    (i * input_size / inner_dim_size);
        out_data[out_index] = input_data[index];
        out_index++;
      }
    }
  }
}

template <class T = float, class IndexType = int64_t, class AxisType = int32_t>
class GatherComputeTest : public arena::TestCase {
 protected:
  // common attributes for this op.
  std::string op_type_ = "gather";
  std::string x_ = "x";
  std::string index_ = "index";
  std::string axis_tensor_ = "Axis";
  std::string out_ = "out";
  DDim x_dims_{{5, 4, 2, 3}};
  DDim index_dims_{{2, 1}};
  DDim axis_dims_{{1}};
  int axis_{0};
  bool is_use_axis_tensor_{false};

 public:
  GatherComputeTest(const Place& place,
                    const std::string& alias,
                    const DDim& x_dims,
                    const DDim& index_dims,
                    const DDim& axis_dims,
                    const bool use_axis_tensor,
                    const int axis_data)
      : TestCase(place, alias),
        x_dims_(x_dims),
        index_dims_(index_dims),
        axis_dims_(axis_dims),
        is_use_axis_tensor_(use_axis_tensor),
        axis_(axis_data) {}

  void RunBaseline(Scope* scope) override {
    operators::GatherParam param;
    auto x = scope->FindTensor(x_);
    auto index = scope->FindTensor(index_);
    param.X = x;
    param.Index = index;
    param.axis = axis_;
    if (is_use_axis_tensor_) {
      param.Axis = scope->FindTensor(axis_tensor_);
    }

    auto index_dims = param.Index->dims();
    auto axis = param.axis;
    auto input_dim = param.X->dims();
    DDim output_dims(input_dim);
    if (param.Axis != nullptr || axis == 0) {
      // if has Axis, we can not obtain correct shape of output
      int batch_size = index_dims[0];
      output_dims[0] = batch_size;
    } else {
      int index_size = index_dims[0];
      std::vector<int64_t> out_dim_vec;
      for (int i = 0; i < axis; i++) {
        out_dim_vec.push_back(input_dim[i]);
      }
      out_dim_vec.push_back(index_size);
      for (int i = axis + 1; i < input_dim.size(); i++) {
        out_dim_vec.push_back(input_dim[i]);
      }
      output_dims = DDim(out_dim_vec);
    }
    auto out = scope->NewTensor(out_);
    out->Resize(output_dims);
    param.Out = out;

    // get axis from tensor
    if (param.Axis != nullptr) {
      LOG(INFO) << "param.Axis is not nullptr";
      const Tensor* axis_tensor = param.Axis;
      const auto& axis_type = axis_tensor->precision();
      if (axis_type == PRECISION(kInt32)) {
        axis = static_cast<int>(axis_tensor->data<int32_t>()[0]);
      } else if (axis_type == PRECISION(kInt64)) {
        axis = static_cast<int>(axis_tensor->data<int64_t>()[0]);
      } else {
        LOG(FATAL) << "unsupport data type of Axis tensor: "
                   << lite_api::PrecisionToStr(axis_type);
      }
    }

    LOG(INFO) << "axis: " << axis;
    LOG(INFO) << "in dims: " << param.X->dims();
    LOG(INFO) << "index dims: " << param.Index->dims();
    LOG(INFO) << "out dims: " << param.Out->dims();

    const auto& data_type = param.X->precision();
    if (axis != 0) {
      switch (data_type) {
        case PRECISION(kFloat):
          GatherV2Func<IndexType, AxisType, float>(param, axis);
          break;
#ifdef ENABLE_ARM_FP16
        case PRECISION(kFP16):
          GatherV2Func<IndexType, AxisType, lite_api::float16_t>(param, axis);
          break;
#endif
        case PRECISION(kInt8):
          GatherV2Func<IndexType, AxisType, int8_t>(param, axis);
          break;
        case PRECISION(kInt16):
          GatherV2Func<IndexType, AxisType, int16_t>(param, axis);
          break;
        case PRECISION(kInt32):
          GatherV2Func<IndexType, AxisType, int32_t>(param, axis);
          break;
        case PRECISION(kInt64):
          GatherV2Func<IndexType, AxisType, int64_t>(param, axis);
          break;
        default:
          LOG(FATAL) << "unsupport data type: "
                     << lite_api::PrecisionToStr(data_type);
      }
      return;
    }

    if (param.X->numel() == 0) return;
    switch (data_type) {
      case PRECISION(kFloat):
        GatherFunc<IndexType, float>(param);
        break;
#ifdef ENABLE_ARM_FP16
      case PRECISION(kFP16):
        GatherFunc<IndexType, lite_api::float16_t>(param);
        break;
#endif
      case PRECISION(kInt8):
        GatherFunc<IndexType, int8_t>(param);
        break;
      case PRECISION(kInt16):
        GatherFunc<IndexType, int16_t>(param);
        break;
      case PRECISION(kInt32):
        GatherFunc<IndexType, int32_t>(param);
        break;
      case PRECISION(kInt64):
        GatherFunc<IndexType, int64_t>(param);
        break;
      default:
        LOG(FATAL) << "unsupport data type: "
                   << lite_api::PrecisionToStr(data_type);
    }
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType(op_type_);
    op_desc->SetInput("X", {x_});
    op_desc->SetInput("Index", {index_});
    if (is_use_axis_tensor_) {
      op_desc->SetInput("Axis", {axis_tensor_});
    }
    op_desc->SetAttr("axis", axis_);
    op_desc->SetOutput("Out", {out_});
  }

  void PrepareData() override {
    std::vector<T> x(x_dims_.production());
    fill_data_rand(
        x.data(), static_cast<T>(-1), static_cast<T>(1), x_dims_.production());

    std::vector<IndexType> index(index_dims_.production());
    fill_data_rand<IndexType>(
        index.data(), 0, x_dims_[0] - 1, index_dims_.production());
    std::vector<AxisType> axis(axis_dims_.production());
    if (is_use_axis_tensor_) {
      fill_data_rand<AxisType>(
          axis.data(), 0, x_dims_.size() - 1, axis_dims_.production());
    }
    SetCommonTensor(x_, x_dims_, x.data());
    SetCommonTensor(index_, index_dims_, index.data());
    if (is_use_axis_tensor_) {
      SetCommonTensor(axis_tensor_, axis_dims_, axis.data());
    }
  }
};

template <class T = float, class IndexType = int32_t, class AxisType = int32_t>
void TestGather(const std::vector<int64_t>& x_dims,
                const std::vector<int64_t>& index_dims,
                const std::vector<int64_t>& axis_dims,
                const bool use_axis_tensor,
                const int axis,
                Place place,
                float abs_error = 1e-5,
                const std::string& alias = "def") {
  std::unique_ptr<arena::TestCase> tester(
      new GatherComputeTest<T, IndexType, AxisType>(place,
                                                    alias,
                                                    DDim(x_dims),
                                                    DDim(index_dims),
                                                    DDim(axis_dims),
                                                    use_axis_tensor,
                                                    axis));
  arena::Arena arena(std::move(tester), place, abs_error);
  arena.TestPrecision();
}

TEST(Gather, precision) {
  float abs_error = 1e-5;
  Place place;
#if defined(LITE_WITH_NPU)
  place = TARGET(kNPU);
  abs_error = 1e-2;  // use fp16 in npu
  // TODO(zhupengyang): enable later
  return;
#elif defined(LITE_WITH_HUAWEI_ASCEND_NPU)
  place = TARGET(kHuaweiAscendNPU);
  abs_error = 1e-2;  // precision_mode default is force_fp16
#elif defined(LITE_WITH_XPU) && defined(LITE_WITH_XTCL)
  place = TARGET(kXPU);
  abs_error = 1e-2;  // use fp16 in xpu
  // TODO(shentanyue): enable later
  return;
#elif defined(LITE_WITH_ARM)
  place = TARGET(kARM);
#elif defined(LITE_WITH_X86)
  place = TARGET(kX86);
#else
  return;
#endif

  for (auto x_dims : std::vector<std::vector<int64_t>>{{5, 7, 10, 12}}) {
    for (auto index_dims : std::vector<std::vector<int64_t>>{{3}, {7}, {10}}) {
      for (auto has_axis_tensor : {false, true}) {
        lite_api::shape_t axis_dims{{1}};
        for (auto axis : {0, 1, 2}) {
#if ((defined(LITE_WITH_XPU) && defined(LITE_WITH_XTCL)) || \
     defined(LITE_WITH_NPU) || defined(LITE_WITH_HUAWEI_ASCEND_NPU))
          TestGather<float, int32_t, int32_t>(x_dims,
                                              index_dims,
                                              axis_dims,
                                              has_axis_tensor,
                                              axis,
                                              place,
                                              abs_error,
                                              "def");
#else
          TestGather<float, int64_t, int64_t>(x_dims,
                                              index_dims,
                                              axis_dims,
                                              has_axis_tensor,
                                              axis,
                                              place,
                                              abs_error,
                                              "int64int64");
          TestGather<float, int32_t, int32_t>(x_dims,
                                              index_dims,
                                              axis_dims,
                                              has_axis_tensor,
                                              axis,
                                              place,
                                              abs_error,
                                              "int32int32");
          TestGather<float, int32_t, int64_t>(x_dims,
                                              index_dims,
                                              axis_dims,
                                              has_axis_tensor,
                                              axis,
                                              place,
                                              abs_error,
                                              "int32int64");
          TestGather<float, int64_t, int32_t>(x_dims,
                                              index_dims,
                                              axis_dims,
                                              has_axis_tensor,
                                              axis,
                                              place,
                                              abs_error,
                                              "int64int32");
          TestGather<int64_t, int64_t, int64_t>(x_dims,
                                                index_dims,
                                                axis_dims,
                                                has_axis_tensor,
                                                axis,
                                                place,
                                                abs_error,
                                                "int64int64");
          TestGather<int64_t, int32_t, int32_t>(x_dims,
                                                index_dims,
                                                axis_dims,
                                                has_axis_tensor,
                                                axis,
                                                place,
                                                abs_error,
                                                "int32int32");
          TestGather<int64_t, int32_t, int64_t>(x_dims,
                                                index_dims,
                                                axis_dims,
                                                has_axis_tensor,
                                                axis,
                                                place,
                                                abs_error,
                                                "int32int64");
          TestGather<int64_t, int64_t, int32_t>(x_dims,
                                                index_dims,
                                                axis_dims,
                                                has_axis_tensor,
                                                axis,
                                                place,
                                                abs_error,
                                                "int64int32");
#endif
        }
      }
    }
  }
}

}  // namespace lite
}  // namespace paddle
