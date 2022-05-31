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
#include "lite/kernels/arm/slice_compute.h"

#include <algorithm>
#include <vector>

#include "lite/backends/arm/math/funcs.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

void DealTensorArray(const operators::SliceParam& param,
                     const std::vector<int64_t>& starts,
                     const std::vector<int64_t>& ends,
                     bool out_is_array) {
  auto in_array = param.XTensorList;
  // If the input is LoDTensorArray, the rank of input is 1.
  int64_t in_size = in_array->size();
  int64_t start = starts[0] < 0 ? (starts[0] + in_size) : starts[0];
  int64_t end = ends[0] < 0 ? (ends[0] + in_size) : ends[0];

  start = std::max(start, static_cast<int64_t>(0));
  end = std::max(end, static_cast<int64_t>(0));
  end = std::min(end, in_size);

  CHECK_GT(end, start) << "end should greater than start";
  int64_t out_size = end - start;

  if (out_is_array) {
    auto out_array = param.OutTensorList;
    out_array->resize(out_size);
    for (int i = 0; i < out_size; ++i) {
      auto* out_tensor = &out_array->at(i);
      auto in_tensor = in_array->at(i + start);
      out_tensor->set_lod(in_tensor.lod());
      if (in_tensor.memory_size() > 0) {
        out_tensor->CopyDataFrom(in_tensor);
      } else {
        VLOG(4) << "WARNING: The input tensor 'x_tensor' holds no memory, so "
                   "nothing has been written to output array["
                << i << "].";
      }
    }
  } else {
    auto out_tensor = param.Out;
    auto in_tensor = in_array->at(start);
    out_tensor->CopyDataFrom(in_tensor);
  }
}

inline std::vector<int64_t> get_new_data_from_tensorlist(
    const std::vector<lite::Tensor*>& list_new_data_tensor) {
  // get tensor
  std::vector<int64_t> vec_new_data;
  for (size_t i = 0; i < list_new_data_tensor.size(); ++i) {
    auto tensor = list_new_data_tensor[i];
    CHECK_EQ(tensor->dims(), DDim({1})) << "shape of dim tensor should be [1]";
    if (tensor->precision() == PrecisionType::kInt32) {
      vec_new_data.push_back(static_cast<int64_t>(*tensor->data<int32_t>()));
    } else if (tensor->precision() == PrecisionType::kInt64) {
      vec_new_data.push_back(static_cast<int64_t>(*tensor->data<int64_t>()));
    } else {
      vec_new_data.push_back(static_cast<int64_t>(*tensor->data<int32_t>()));
      LOG(WARNING) << "slice StartsTensor or EndsTensor :The dtype of Tensor "
                      "must be int32 "
                      "or int64";
    }
  }
  return vec_new_data;
}

inline std::vector<int64_t> get_new_data_from_tensor(
    const lite::Tensor* new_data_tensor) {
  std::vector<int64_t> vec_new_data;
  if (new_data_tensor->precision() == PrecisionType::kInt32) {
    auto* new_data = new_data_tensor->data<int32_t>();
    // a int32->int64 convert here
    vec_new_data =
        std::vector<int64_t>(new_data, new_data + new_data_tensor->numel());
  } else if (new_data_tensor->precision() == PrecisionType::kInt64) {
    auto* new_data = new_data_tensor->data<int64_t>();
    vec_new_data =
        std::vector<int64_t>(new_data, new_data + new_data_tensor->numel());
  } else {
    auto* new_data = new_data_tensor->data<int32_t>();
    vec_new_data =
        std::vector<int64_t>(new_data, new_data + new_data_tensor->numel());
    LOG(WARNING)
        << "slice StartsTensor or EndsTensor :The dtype of Tensor must "
           "be int32 "
           "or int64";
  }
  return vec_new_data;
}

template <typename T, PrecisionType PType>
void SliceCompute<T, PType>::Run() {
  auto& ctx = this->ctx_->template As<ARMContext>();
  auto& param = this->template Param<operators::SliceParam>();
  std::vector<int> axes = param.axes;
  std::vector<int32_t> starts_int = param.starts;
  std::vector<int32_t> ends_int = param.ends;
  std::vector<int64_t> starts(starts_int.begin(), starts_int.end());
  std::vector<int64_t> ends(ends_int.begin(), ends_int.end());
  std::vector<int> decrease_axis = param.decrease_axis;
  std::vector<int> infer_flags = param.infer_flags;

  auto list_new_ends_tensor = param.EndsTensorList;
  auto list_new_starts_tensor = param.StartsTensorList;
  bool need_infer = false;
  if (param.StartsTensor || param.EndsTensor) {
    need_infer = true;
  }
  if (list_new_starts_tensor.size() > 0 || list_new_ends_tensor.size() > 0) {
    need_infer = true;
  }
  if (need_infer) {
    if (param.StartsTensor) {
      starts = get_new_data_from_tensor(param.StartsTensor);
    } else if (list_new_starts_tensor.size() > 0) {
      starts = get_new_data_from_tensorlist(list_new_starts_tensor);
    }
    CHECK_EQ(starts.size(), axes.size())
        << "The size of starts must be equal to the size of axes.";
    if (param.EndsTensor) {
      ends = get_new_data_from_tensor(param.EndsTensor);
    } else if (list_new_ends_tensor.size() > 0) {
      ends = get_new_data_from_tensorlist(list_new_ends_tensor);
    }
    CHECK_EQ(ends.size(), axes.size())
        << "The size of ends must be equal to the size of axes.";
  }
  // if slice input is tensor_array
  if (param.X == nullptr && param.XTensorList != nullptr) {
    DealTensorArray(param,
                    starts,
                    ends,
                    (param.Out == nullptr && param.OutTensorList != nullptr));
    return;
  }
  int64_t dim_value, start, end;
  auto in = param.X;
  auto in_dims = in->dims();
  auto out = param.Out;
  auto out_dims = out->dims();
  out_dims = in_dims;
  for (size_t i = 0; i < axes.size(); ++i) {
    dim_value = out_dims[axes[i]];
    if (dim_value > 0) {
      // when end = start+1 and start == -1
      if (starts[i] == -1 && ends[i] == 0 && infer_flags[i] == -1) {
        auto ret =
            std::find(decrease_axis.begin(), decrease_axis.end(), axes[i]);
        if (ret != decrease_axis.end()) {
          ends[i] = 10000000;
        }
      }

      start = starts[i] < 0 ? (starts[i] + dim_value) : starts[i];
      end = ends[i] < 0 ? (ends[i] + dim_value) : ends[i];
      start = std::max(start, static_cast<int64_t>(0));
      end = std::max(end, static_cast<int64_t>(0));
      end = std::min(end, dim_value);
      CHECK_GT(end, start) << "end should greater than start";
      out_dims[axes[i]] = end - start;
    }
  }
  // generate new shape
  if (decrease_axis.size() > 0) {
    std::vector<int64_t> new_out_shape;
    for (size_t i = 0; i < decrease_axis.size(); ++i) {
      CHECK_EQ(out_dims[decrease_axis[i]], 1) << "decrease dim should be 1";
      out_dims[decrease_axis[i]] = 0;
    }

    for (int i = 0; i < out_dims.size(); ++i) {
      if (out_dims[i] != 0) {
        new_out_shape.push_back(out_dims[i]);
      }
    }
    if (new_out_shape.size() == 0) {
      new_out_shape.push_back(1);
    }
    DDim new_dims;
    new_dims.ConstructFrom(new_out_shape);
    out_dims = new_dims;
  }
  out->Resize(out_dims);
  const auto* x_data = in->template data<T>();
  auto* o_data = out->template mutable_data<T>();
  std::vector<int32_t> starts_final(starts.begin(), starts.end());
  std::vector<int32_t> ends_final(ends.begin(), ends.end());
  lite::arm::math::slice(
      x_data, in_dims.data(), axes, starts_final, ends_final, o_data, &ctx);
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

using slice_float =
    paddle::lite::kernels::arm::SliceCompute<float, PRECISION(kFloat)>;
REGISTER_LITE_KERNEL(slice, kARM, kFloat, kNCHW, slice_float, def)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFloat))})
    .BindInput("StartsTensor",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .BindInput("EndsTensor",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .BindInput("StartsTensorList",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .BindInput("EndsTensorList",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFloat))})
    .Finalize();

REGISTER_LITE_KERNEL(slice, kARM, kFloat, kNCHW, slice_float, array_def)
    .BindInput("Input",
               {LiteType::GetTensorListTy(TARGET(kARM), PRECISION(kFloat))})
    .BindInput("StartsTensor",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .BindInput("EndsTensor",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .BindInput("StartsTensorList",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .BindInput("EndsTensorList",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFloat))})
    .Finalize();

REGISTER_LITE_KERNEL(
    slice, kARM, kFloat, kNCHW, slice_float, float_i64_starts_ends)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFloat))})
    .BindInput("StartsTensor",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt64))})
    .BindInput("EndsTensor",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt64))})
    .BindInput("StartsTensorList",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt64))})
    .BindInput("EndsTensorList",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt64))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFloat))})
    .Finalize();

REGISTER_LITE_KERNEL(
    slice, kARM, kFloat, kNCHW, slice_float, array_float_i64_starts_ends)
    .BindInput("Input",
               {LiteType::GetTensorListTy(TARGET(kARM), PRECISION(kFloat))})
    .BindInput("StartsTensor",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt64))})
    .BindInput("EndsTensor",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt64))})
    .BindInput("StartsTensorList",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt64))})
    .BindInput("EndsTensorList",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt64))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFloat))})
    .Finalize();

using slice_boolean =
    paddle::lite::kernels::arm::SliceCompute<bool, PRECISION(kFloat)>;
REGISTER_LITE_KERNEL(slice, kARM, kFloat, kNCHW, slice_boolean, bool_slice)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kBool))})
    .BindInput("StartsTensor",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .BindInput("EndsTensor",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .BindInput("StartsTensorList",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .BindInput("EndsTensorList",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kBool))})
    .Finalize();

REGISTER_LITE_KERNEL(
    slice, kARM, kFloat, kNCHW, slice_boolean, array_bool_slice)
    .BindInput("Input",
               {LiteType::GetTensorListTy(TARGET(kARM), PRECISION(kBool))})
    .BindInput("StartsTensor",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .BindInput("EndsTensor",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .BindInput("StartsTensorList",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .BindInput("EndsTensorList",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kBool))})
    .Finalize();

using slice_int32 =
    paddle::lite::kernels::arm::SliceCompute<int, PRECISION(kFloat)>;
REGISTER_LITE_KERNEL(slice, kARM, kFloat, kNCHW, slice_int32, int32_slice)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .BindInput("StartsTensor",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt64))})
    .BindInput("EndsTensor",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt64))})
    .BindInput("StartsTensorList",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt64))})
    .BindInput("EndsTensorList",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt64))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .Finalize();

REGISTER_LITE_KERNEL(slice, kARM, kFloat, kNCHW, slice_int32, array_int32_slice)
    .BindInput("Input",
               {LiteType::GetTensorListTy(TARGET(kARM), PRECISION(kInt32))})
    .BindInput("StartsTensor",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt64))})
    .BindInput("EndsTensor",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt64))})
    .BindInput("StartsTensorList",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt64))})
    .BindInput("EndsTensorList",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt64))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .Finalize();

using slice_int64 =
    paddle::lite::kernels::arm::SliceCompute<int64_t, PRECISION(kFloat)>;

REGISTER_LITE_KERNEL(slice, kARM, kFloat, kNCHW, slice_int64, def_int64)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt64))})
    .BindInput("StartsTensor",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt64))})
    .BindInput("EndsTensor",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt64))})
    .BindInput("StartsTensorList",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt64))})
    .BindInput("EndsTensorList",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt64))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt64))})
    .Finalize();

REGISTER_LITE_KERNEL(slice, kARM, kFloat, kNCHW, slice_int64, array_def_int64)
    .BindInput("Input",
               {LiteType::GetTensorListTy(TARGET(kARM), PRECISION(kInt64))})
    .BindInput("StartsTensor",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt64))})
    .BindInput("EndsTensor",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt64))})
    .BindInput("StartsTensorList",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt64))})
    .BindInput("EndsTensorList",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt64))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt64))})
    .Finalize();
