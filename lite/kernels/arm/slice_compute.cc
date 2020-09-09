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

inline std::vector<int32_t> get_new_data_from_tensorlist(
    const std::vector<lite::Tensor*>& list_new_data_tensor) {
  // get tensor
  std::vector<int32_t> vec_new_data;
  for (size_t i = 0; i < list_new_data_tensor.size(); ++i) {
    auto tensor = list_new_data_tensor[i];
    CHECK_EQ(tensor->dims(), DDim({1})) << "shape of dim tensor should be [1]";
    vec_new_data.push_back(static_cast<int32_t>(*tensor->data<int32_t>()));
  }
  return vec_new_data;
}

inline std::vector<int32_t> get_new_data_from_tensor(
    const lite::Tensor* new_data_tensor) {
  std::vector<int32_t> vec_new_data;
  auto* new_data = new_data_tensor->data<int32_t>();
  vec_new_data =
      std::vector<int32_t>(new_data, new_data + new_data_tensor->numel());
  return vec_new_data;
}

template <typename T, PrecisionType PType>
void SliceCompute<T, PType>::Run() {
  auto& ctx = this->ctx_->template As<ARMContext>();
  auto& param = this->template Param<operators::SliceParam>();

  auto in = param.X;
  auto in_dims = in->dims();
  auto out = param.Out;
  auto out_dims = out->dims();

  std::vector<int> axes = param.axes;
  std::vector<int32_t> starts = param.starts;
  std::vector<int32_t> ends = param.ends;
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
    out_dims = in_dims;
    int dim_value, start, end;
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
        start = std::max(start, 0);
        end = std::max(end, 0);
        end = std::min(end, dim_value);
        CHECK_GT(end, start) << "end should greater than start";
        out_dims[axes[i]] = end - start;
      }
    }
    out->Resize(out_dims);
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
  }

  // resize out dims
  if (decrease_axis.size() > 0) {
    if (decrease_axis.size() == (size_t)in_dims.size()) {
      std::vector<int64_t> vec_origin_out_shape(decrease_axis.size(), 1);
      out->Resize(DDim(vec_origin_out_shape));
    } else {
      std::vector<int64_t> vec_origin_out_shape(
          out_dims.size() + decrease_axis.size(), -1);

      for (size_t i = 0; i < decrease_axis.size(); ++i) {
        vec_origin_out_shape[decrease_axis[i]] = 1;
      }

      int index = 0;
      for (size_t i = 0; i < vec_origin_out_shape.size(); ++i) {
        if (vec_origin_out_shape[i] == -1) {
          vec_origin_out_shape[i] = out_dims[index];
          ++index;
        }
      }

      out->Resize(DDim(vec_origin_out_shape));
    }
  }

  auto new_out_dims = out->dims();
  const auto* x_data = in->template data<T>();
  auto* o_data = out->template mutable_data<T>();
  lite::arm::math::slice(
      x_data, in_dims.data(), axes, starts, ends, o_data, &ctx);
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
               {LiteType::GetTensorListTy(TARGET(kARM), PRECISION(kInt32))})
    .BindInput("EndsTensorList",
               {LiteType::GetTensorListTy(TARGET(kARM), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFloat))})
    .Finalize();

using slice_int32 =
    paddle::lite::kernels::arm::SliceCompute<int, PRECISION(kInt32)>;
REGISTER_LITE_KERNEL(slice, kARM, kInt32, kNCHW, slice_int32, def)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .BindInput("StartsTensor",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .BindInput("EndsTensor",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .BindInput("StartsTensorList",
               {LiteType::GetTensorListTy(TARGET(kARM), PRECISION(kInt32))})
    .BindInput("EndsTensorList",
               {LiteType::GetTensorListTy(TARGET(kARM), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .Finalize();

using slice_int64 =
    paddle::lite::kernels::arm::SliceCompute<int64_t, PRECISION(kInt64)>;
REGISTER_LITE_KERNEL(slice, kARM, kInt64, kNCHW, slice_int64, def)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt64))})
    .BindInput("StartsTensor",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .BindInput("EndsTensor",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .BindInput("StartsTensorList",
               {LiteType::GetTensorListTy(TARGET(kARM), PRECISION(kInt32))})
    .BindInput("EndsTensorList",
               {LiteType::GetTensorListTy(TARGET(kARM), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt64))})
    .Finalize();
