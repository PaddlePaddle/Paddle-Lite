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

#include "lite/kernels/xpu/elementwise_compute.h"
#include <algorithm>
#include <functional>
#include <string>
#include <vector>
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_lite.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

inline DDim TrimTrailingSingularDims(const DDim& dims) {
  // Remove trailing dimensions of size 1 for y
  auto actual_dims_size = dims.size();
  for (; actual_dims_size != 0; --actual_dims_size) {
    if (dims[actual_dims_size - 1] != 1) break;
  }

  std::vector<int64_t> trim_dims;
  trim_dims.resize(actual_dims_size);
  for (int i = 0; i < actual_dims_size; ++i) {
    trim_dims[i] = dims[i];
  }
  if (trim_dims.size() == 0) {
    return DDim();
  }
  DDim actual_dims = DDim(trim_dims);
  return actual_dims;
}

inline void GetMidDims(const DDim& x_dims,
                       const DDim& y_dims,
                       const int axis,
                       int* pre,
                       int* n,
                       int* post,
                       int* mid_flag = NULL) {
  *pre = 1;
  *n = 1;
  *post = 1;
  if (mid_flag != NULL) {
    *mid_flag = 0;
    int mid = 0;
    for (int i = 0; i < axis; ++i) {
      (*pre) *= x_dims[i];
    }
    for (int i = 0; i < y_dims.size(); ++i) {
      if (x_dims[i + axis] != y_dims[i]) {
        // only support single y_dims[i] = 1 now.
        CHECK_EQ(*mid_flag, 0) << "Broadcast support y_dims with single 1.";
        CHECK_EQ(y_dims[i], 1) << "Broadcast dimension mismatch.";
        // m*n*k m*1*k
        for (int j = 0; j < i; ++j) {
          (*pre) *= y_dims[j];
        }
        *n = std::max(x_dims[i + axis], y_dims[i]);
        *mid_flag = 1;
        mid = i;
        break;
      }
      (*n) *= y_dims[i];
    }
    if (*mid_flag) {
      for (int i = mid + 1; i < x_dims.size(); ++i) {
        (*post) *= x_dims[i];
      }
    } else {
      for (int i = axis + y_dims.size(); i < x_dims.size(); ++i) {
        (*post) *= x_dims[i];
      }
    }
  } else {
    for (int i = 0; i < axis; ++i) {
      (*pre) *= x_dims[i];
    }

    for (int i = 0; i < y_dims.size(); ++i) {
      CHECK_EQ(x_dims[i + axis], y_dims[i]) << "Broadcast dimension mismatch.";
      (*n) *= y_dims[i];
    }

    for (int i = axis + y_dims.size(); i < x_dims.size(); ++i) {
      (*post) *= x_dims[i];
    }
  }
}

void ElementwiseAddCompute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->As<XPUContext>();

  auto& x_dims = param.X->dims();
  auto& y_dims = param.Y->dims();
  int axis = param.axis;

  auto y_dims_untrimed = y_dims;
  axis = (axis == -1 ? x_dims.size() - y_dims_untrimed.size() : axis);
  auto y_dims_after_trailing = TrimTrailingSingularDims(y_dims_untrimed);
  axis = (y_dims_after_trailing.size() == 0) ? x_dims.size() : axis;
  int pre, n, post;
  GetMidDims(x_dims, y_dims_after_trailing, axis, &pre, &n, &post);
  int len = pre * n * post;
  float* y_broadcast = nullptr;

  if (post == 1) {
    int r =
        xdnn::matrix_vector_add(ctx.GetRawContext(),
                                param.X->data<float>(),
                                param.Y->data<float>(),
                                param.Out->mutable_data<float>(TARGET(kXPU)),
                                pre,
                                n);
    CHECK_EQ(r, 0);
    return;
  }
  if (pre != 1 || post != 1) {
    XPUScratchPadGuard y_broadcast_xpu_guard_ =
        TargetWrapperXPU::MallocScratchPad(len * sizeof(float),
                                           false /* use_l3 */);
    y_broadcast = reinterpret_cast<float*>(y_broadcast_xpu_guard_->addr_);

    int r = xdnn::broadcast_ew(ctx.GetRawContext(),
                               param.Y->data<float>(),
                               y_broadcast,
                               pre,
                               n,
                               post,
                               xdnn::ElementwiseOp::ASSIGN);
    CHECK_EQ(r, 0);
    r = xdnn::elementwise_add(
        ctx.GetRawContext(),                          /* context */
        param.X->data<float>(),                       /* x */
        y_broadcast,                                  /* y */
        param.Out->mutable_data<float>(TARGET(kXPU)), /* z */
        len);
    CHECK_EQ(r, 0);
    return;
  }
  int r = xdnn::elementwise_add(
      ctx.GetRawContext(),                          /* context */
      param.X->data<float>(),                       /* x */
      param.Y->data<float>(),                       /* y */
      param.Out->mutable_data<float>(TARGET(kXPU)), /* z */
      len);
  CHECK_EQ(r, 0);
}

void ElementwiseMulCompute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->As<XPUContext>();

  auto& x_dims = param.X->dims();
  auto& y_dims = param.Y->dims();
  int axis = param.axis;

  auto y_dims_untrimed = y_dims;
  axis = (axis == -1 ? x_dims.size() - y_dims_untrimed.size() : axis);
  auto y_dims_after_trailing = TrimTrailingSingularDims(y_dims_untrimed);
  axis = (y_dims_after_trailing.size() == 0) ? x_dims.size() : axis;
  int pre, n, post;
  GetMidDims(x_dims, y_dims_after_trailing, axis, &pre, &n, &post);
  int len = pre * n * post;
  float* y_broadcast = nullptr;

  if (post == 1) {
    int r =
        xdnn::matrix_vector_mul(ctx.GetRawContext(),
                                param.X->data<float>(),
                                param.Y->data<float>(),
                                param.Out->mutable_data<float>(TARGET(kXPU)),
                                pre,
                                n);
    CHECK_EQ(r, 0);
    return;
  }
  if (pre != 1 || post != 1) {
    XPUScratchPadGuard y_broadcast_xpu_guard_ =
        TargetWrapperXPU::MallocScratchPad(len * sizeof(float),
                                           false /* use_l3 */);
    y_broadcast = reinterpret_cast<float*>(y_broadcast_xpu_guard_->addr_);

    int r = xdnn::broadcast_ew(ctx.GetRawContext(),
                               param.Y->data<float>(),
                               y_broadcast,
                               pre,
                               n,
                               post,
                               xdnn::ElementwiseOp::ASSIGN);
    CHECK_EQ(r, 0);
    r = xdnn::elementwise_mul(
        ctx.GetRawContext(),                          /* context */
        param.X->data<float>(),                       /* x */
        y_broadcast,                                  /* y */
        param.Out->mutable_data<float>(TARGET(kXPU)), /* z */
        len);
    CHECK_EQ(r, 0);
    return;
  }
  int r = xdnn::elementwise_mul(
      ctx.GetRawContext(),                          /* context */
      param.X->data<float>(),                       /* x */
      param.Y->data<float>(),                       /* y */
      param.Out->mutable_data<float>(TARGET(kXPU)), /* z */
      len);
  CHECK_EQ(r, 0);
}

void ElementwiseSubCompute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->As<XPUContext>();

  auto& x_dims = param.X->dims();
  auto& y_dims = param.Y->dims();
  int axis = param.axis;

  auto y_dims_untrimed = y_dims;
  axis = (axis == -1 ? x_dims.size() - y_dims_untrimed.size() : axis);
  auto y_dims_after_trailing = TrimTrailingSingularDims(y_dims_untrimed);
  axis = (y_dims_after_trailing.size() == 0) ? x_dims.size() : axis;
  int pre, n, post;
  GetMidDims(x_dims, y_dims_after_trailing, axis, &pre, &n, &post);
  int len = pre * n * post;
  float* y_broadcast = nullptr;

  if (len != param.Y->numel()) {
    XPUScratchPadGuard y_broadcast_xpu_guard_ =
        TargetWrapperXPU::MallocScratchPad(len * sizeof(float),
                                           false /* use_l3 */);
    y_broadcast = reinterpret_cast<float*>(y_broadcast_xpu_guard_->addr_);

    int r = xdnn::broadcast_ew(ctx.GetRawContext(),
                               param.Y->data<float>(),
                               y_broadcast,
                               pre,
                               n,
                               post,
                               xdnn::ElementwiseOp::ASSIGN);
    CHECK_EQ(r, 0);
    r = xdnn::elementwise_sub(
        ctx.GetRawContext(),                          /* context */
        param.X->data<float>(),                       /* x */
        y_broadcast,                                  /* y */
        param.Out->mutable_data<float>(TARGET(kXPU)), /* z */
        len);
    CHECK_EQ(r, 0);
    return;
  }
  int r = xdnn::elementwise_sub(
      ctx.GetRawContext(),                          /* context */
      param.X->data<float>(),                       /* x */
      param.Y->data<float>(),                       /* y */
      param.Out->mutable_data<float>(TARGET(kXPU)), /* z */
      len);
  CHECK_EQ(r, 0);
}

void ElementwiseDivCompute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->As<XPUContext>();

  auto& x_dims = param.X->dims();
  auto& y_dims = param.Y->dims();
  int axis = param.axis;

  auto y_dims_untrimed = y_dims;
  axis = (axis == -1 ? x_dims.size() - y_dims_untrimed.size() : axis);
  auto y_dims_after_trailing = TrimTrailingSingularDims(y_dims_untrimed);
  axis = (y_dims_after_trailing.size() == 0) ? x_dims.size() : axis;
  int pre, n, post;
  GetMidDims(x_dims, y_dims_after_trailing, axis, &pre, &n, &post);
  int len = pre * n * post;
  float* y_broadcast = nullptr;

  if (len != param.Y->numel()) {
    XPUScratchPadGuard y_broadcast_xpu_guard_ =
        TargetWrapperXPU::MallocScratchPad(len * sizeof(float),
                                           false /* use_l3 */);
    y_broadcast = reinterpret_cast<float*>(y_broadcast_xpu_guard_->addr_);

    int r = xdnn::broadcast_ew(ctx.GetRawContext(),
                               param.Y->data<float>(),
                               y_broadcast,
                               pre,
                               n,
                               post,
                               xdnn::ElementwiseOp::ASSIGN);
    CHECK_EQ(r, 0);
    r = xdnn::elementwise_div(
        ctx.GetRawContext(),                          /* context */
        param.X->data<float>(),                       /* x */
        y_broadcast,                                  /* y */
        param.Out->mutable_data<float>(TARGET(kXPU)), /* z */
        len);
    CHECK_EQ(r, 0);
    return;
  }
  int r = xdnn::elementwise_div(
      ctx.GetRawContext(),                          /* context */
      param.X->data<float>(),                       /* x */
      param.Y->data<float>(),                       /* y */
      param.Out->mutable_data<float>(TARGET(kXPU)), /* z */
      len);
  CHECK_EQ(r, 0);
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(elementwise_add,
                     kXPU,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::xpu::ElementwiseAddCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();

REGISTER_LITE_KERNEL(elementwise_mul,
                     kXPU,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::xpu::ElementwiseMulCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();

REGISTER_LITE_KERNEL(elementwise_sub,
                     kXPU,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::xpu::ElementwiseSubCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();

REGISTER_LITE_KERNEL(elementwise_div,
                     kXPU,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::xpu::ElementwiseDivCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();
