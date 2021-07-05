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

#include "lite/kernels/xpu/slice_compute.h"
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite_metal {
namespace kernels {
namespace xpu {

inline std::vector<int> GetIntDataFromTensorList(
    const std::vector<lite_metal::Tensor*>& list_tensor) {
  std::vector<int> vec_data;
  for (auto& tensor_i : list_tensor) {
    CHECK_EQ(tensor_i->dims(), DDim({1}))
        << "shape of dim tensor should be [1]";
    auto precision = tensor_i->precision();
    switch (precision) {
      case PRECISION(kInt32): {
        vec_data.push_back(*tensor_i->data<int>());
        break;
      }
      case PRECISION(kInt64): {
        vec_data.push_back(static_cast<int>(*tensor_i->data<int64_t>()));
        break;
      }
      default: {
        LOG(FATAL) << "unsupported data precision: "
                   << lite_metal_api::PrecisionToStr(precision);
        break;
      }
    }
  }
  return vec_data;
}

inline std::vector<int> GetIntDataFromTensor(const Tensor* tensor) {
  std::vector<int> vec_data;
  auto precision = tensor->precision();
  switch (precision) {
    case PRECISION(kInt32): {
      const int* data = tensor->data<int>();
      vec_data = std::vector<int>(data, data + tensor->numel());
      break;
    }
    case PRECISION(kInt64): {
      const int64_t* data = tensor->data<int64_t>();
      for (int64_t i = 0; i < tensor->numel(); i++) {
        vec_data.push_back(static_cast<int>(data[i]));
      }
      break;
    }
    default: {
      LOG(FATAL) << "unsupported data precision: "
                 << lite_metal_api::PrecisionToStr(precision);
      break;
    }
  }
  return vec_data;
}

template <class T>
void SliceCompute<T>::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();

  auto out = param.Out;
  auto in = param.X;
  auto axes = param.axes;
  auto StartsTensor = param.StartsTensor;
  auto EndsTensor = param.EndsTensor;
  auto StartsTensorList = param.StartsTensorList;
  auto EndsTensorList = param.EndsTensorList;
  auto starts = param.starts;
  auto ends = param.ends;
  auto infer_flags = param.infer_flags;
  auto decrease_axis = param.decrease_axis;

  auto out_dims = out->dims();
  auto in_dims = in->dims();

  bool need_infer = false;
  if (StartsTensor || EndsTensor) {
    need_infer = true;
  } else if (StartsTensorList.size() > 0 || EndsTensorList.size() > 0) {
    need_infer = true;
  }

  if (need_infer) {
    if (StartsTensor) {
      starts = GetIntDataFromTensor(StartsTensor);
    } else if (StartsTensorList.size() > 0) {
      starts = GetIntDataFromTensorList(StartsTensorList);
    }
    CHECK_EQ(starts.size(), axes.size())
        << "The size of starts must be equal to the size of axes.";
    if (EndsTensor) {
      ends = GetIntDataFromTensor(EndsTensor);
    } else if (EndsTensorList.size() > 0) {
      ends = GetIntDataFromTensorList(EndsTensorList);
    }
    CHECK_EQ(ends.size(), axes.size())
        << "The size of ends must be equal to the size of axes.";
    out_dims = in_dims;
    int dim_value, start, end;
    for (size_t i = 0; i < axes.size(); ++i) {
      dim_value = out_dims[axes[i]];
      if (dim_value > 0) {
        // when end = start + 1 and start == -1
        if (starts[i] == -1 && ends[i] == 0 && infer_flags[i] == -1) {
          auto ret =
              std::find(decrease_axis.begin(), decrease_axis.end(), axes[i]);
          if (ret != decrease_axis.end()) {
            ends[i] = 10000000;
          }
        }

        start = starts[i] < 0 ? (starts[i] + dim_value) : starts[i];
        end = ends[i] < 0 ? (ends[i] + dim_value) : ends[i];
        start = (std::max)(start, 0);
        end = (std::max)(end, 0);
        end = (std::min)(end, dim_value);
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

      for (size_t i = 0; i < out_dims.size(); ++i) {
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
  auto x_shape = in_dims.Vectorize();
  std::vector<int> x_shape_(x_shape.begin(), x_shape.end());
  std::vector<int> x_dim_begin_(in_dims.size(), 0);
  std::vector<int> x_dim_end_(x_shape_);

  for (size_t i = 0; i < axes.size(); ++i) {
    int axis = axes[i];
    x_dim_begin_[axis] =
        starts[i] < 0 ? starts[i] + static_cast<int>(in_dims[axis]) : starts[i];
    int end = ends[i] < 0 ? ends[i] + static_cast<int>(in_dims[axis]) : ends[i];
    x_dim_end_[axis] = (std::min)(end, static_cast<int>(in_dims[axis]));
  }

  int r =
      xdnn::slice(ctx.GetRawContext(),         /* context */
                  param.X->template data<T>(), /* in */
                  param.Out->template mutable_data<T>(TARGET(kXPU)), /* out */
                  x_shape_,
                  x_dim_begin_,
                  x_dim_end_);

  CHECK_EQ(r, 0);
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

using SliceFloat32 = paddle::lite_metal::kernels::xpu::SliceCompute<float>;
REGISTER_LITE_KERNEL(slice, kXPU, kFloat, kAny, SliceFloat32, def)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindInput("StartsTensor",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kAny))})
    .BindInput("EndsTensor",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kAny))})
    .BindInput("StartsTensorList",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kAny))})
    .BindInput("EndsTensorList",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kAny))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .Finalize();

using SliceInt32 = paddle::lite_metal::kernels::xpu::SliceCompute<int32_t>;
REGISTER_LITE_KERNEL(slice, kXPU, kFloat, kAny, SliceInt32, int32)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt32))})
    .BindInput("StartsTensor",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kAny))})
    .BindInput("EndsTensor",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kAny))})
    .BindInput("StartsTensorList",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kAny))})
    .BindInput("EndsTensorList",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kAny))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt32))})
    .Finalize();

using SliceInt64 = paddle::lite_metal::kernels::xpu::SliceCompute<int64_t>;
REGISTER_LITE_KERNEL(slice, kXPU, kFloat, kAny, SliceInt64, int64)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt64))})
    .BindInput("StartsTensor",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kAny))})
    .BindInput("EndsTensor",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kAny))})
    .BindInput("StartsTensorList",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kAny))})
    .BindInput("EndsTensorList",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kAny))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt64))})
    .Finalize();
