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

#include "lite/kernels/xpu/reduce_compute.h"
#include <vector>
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

template <typename inType,
          typename outType,
          typename Functor,
          PrecisionType PType>
void ReduceCompute<inType, outType, Functor, PType>::PrepareForRun() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();
  int max_ptr_size = ctx.GetRawContext()->max_ptr_size();
  if (param.enable_int8) {
    quant_x_max_value_guard_ =
        TargetWrapperXPU::MallocScratchPad(max_ptr_size * sizeof(float));
    std::vector<float> cpu_quant_x_max_value(max_ptr_size, param.input_scale);
    lite::TargetWrapperXPU::MemcpySync(quant_x_max_value_guard_->addr_,
                                       cpu_quant_x_max_value.data(),
                                       sizeof(float) * max_ptr_size,
                                       IoDirection::HtoD);
    quant_out_max_value_guard_ =
        TargetWrapperXPU::MallocScratchPad(max_ptr_size * sizeof(float));
    std::vector<float> cpu_quant_out_max_value(max_ptr_size,
                                               param.output_scale);
    lite::TargetWrapperXPU::MemcpySync(quant_out_max_value_guard_->addr_,
                                       cpu_quant_out_max_value.data(),
                                       sizeof(float) * max_ptr_size,
                                       IoDirection::HtoD);
  }
}

template <typename InType, typename OutType>
struct ReduceSumMixFunctor {
  inline int operator()(xdnn::Context* ctx,
                        const InType* x,
                        OutType* out,
                        const std::vector<int>& xshape,
                        const std::vector<int>& dims,
                        const float* max_x,
                        const float* max_y) const {
    int ret = 0;
    XPUScratchPadGuard cast_guard_x = TargetWrapperXPU::MallocScratchPad(1 * 4);
    if (!std::is_same<InType, OutType>::value) {
      int nsize = 1;
      int nsize_x = 1;
      std::vector<int> size_shape(xshape.begin(), xshape.end());
      for (int32_t i = 0; i < dims.size(); ++i) {
        size_shape[dims[i]] = 1;
      }
      for (auto item : size_shape) {
        nsize *= item;
      }
      for (auto item : xshape) {
        nsize_x *= item;
      }

      cast_guard_x->Reserve(nsize_x * sizeof(OutType));
      ret = xdnn::cast_v2<InType, OutType>(
          ctx, x, reinterpret_cast<OutType*>(cast_guard_x->addr_), nsize_x);
      CHECK_EQ(ret, 0);

      ret = xdnn::reduce_sum<OutType>(
          ctx,
          reinterpret_cast<OutType*>(cast_guard_x->addr_),
          out,
          xshape,
          dims);
      CHECK_EQ(ret, 0);
    } else {
      LOG(FATAL) << "outtype and intype should be different.";
    }
    return ret;
  }
};

template <typename T>
struct ReduceSumFunctor {
  inline int operator()(xdnn::Context* ctx,
                        const T* x,
                        T* out,
                        const std::vector<int>& xshape,
                        const std::vector<int>& dims,
                        const float* max_x,
                        const float* max_y) const {
    return xdnn::reduce_sum<T>(ctx, x, out, xshape, dims);
  }
};

template <typename T>
struct ReduceMeanFunctor {
  inline int operator()(xdnn::Context* ctx,
                        const T* x,
                        T* out,
                        const std::vector<int>& xshape,
                        const std::vector<int>& dims,
                        const float* max_x,
                        const float* max_y) const {
    return xdnn::reduce_mean<T>(ctx, x, out, xshape, dims);
  }
};

// ReduceMeanFunctor int8
template <>
struct ReduceMeanFunctor<int8_t> {
  inline int operator()(xdnn::Context* ctx,
                        const int8_t* x,
                        int8_t* out,
                        const std::vector<int>& xshape,
                        const std::vector<int>& dims,
                        const float* max_x,
                        float* max_y) const {
    return xdnn::avg_pool2d<int8_t>(ctx,
                                    x,
                                    out,
                                    xshape[0],
                                    xshape[1],
                                    xshape[2],
                                    xshape[3],
                                    {xshape[2], xshape[3]},
                                    {1},
                                    {0},
                                    true,
                                    true,
                                    max_x,
                                    max_y);
  }
};

template <typename T>
struct ReduceProdFunctor {
  inline int operator()(xdnn::Context* ctx,
                        const T* x,
                        T* out,
                        const std::vector<int>& xshape,
                        const std::vector<int>& dims,
                        const float* max_x,
                        const float* max_y) const {
    return xdnn::reduce_prod<T>(ctx, x, out, xshape, dims);
  }
};

template <typename T>
struct ReduceMinFunctor {
  inline int operator()(xdnn::Context* ctx,
                        const T* x,
                        T* out,
                        const std::vector<int>& xshape,
                        const std::vector<int>& dims,
                        const float* max_x,
                        const float* max_y) const {
    return xdnn::reduce_min<T>(ctx, x, out, xshape, dims);
  }
};

template <typename T>
struct ReduceMaxFunctor {
  inline int operator()(xdnn::Context* ctx,
                        const T* x,
                        T* out,
                        const std::vector<int>& xshape,
                        const std::vector<int>& dims,
                        const float* max_x,
                        const float* max_y) const {
    return xdnn::reduce_max<T>(ctx, x, out, xshape, dims);
  }
};

template <typename T>
struct ReduceAllFunctor {
  inline int operator()(xdnn::Context* ctx,
                        const T* x,
                        T* out,
                        const std::vector<int>& xshape,
                        const std::vector<int>& dims,
                        const float* max_x,
                        const float* max_y) const {
    return xdnn::reduce_all<int8_t>(ctx,
                                    reinterpret_cast<const int8_t*>(x),
                                    reinterpret_cast<int8_t*>(out),
                                    xshape,
                                    dims);
  }
};

template <typename T>
struct ReduceAnyFunctor {
  inline int operator()(xdnn::Context* ctx,
                        const T* x,
                        T* out,
                        const std::vector<int>& xshape,
                        const std::vector<int>& dims,
                        const float* max_x,
                        const float* max_y) const {
    return xdnn::reduce_any<int8_t>(ctx,
                                    reinterpret_cast<const int8_t*>(x),
                                    reinterpret_cast<int8_t*>(out),
                                    xshape,
                                    dims);
  }
};

template <typename inType,
          typename outType,
          typename Functor,
          PrecisionType PType>
void ReduceCompute<inType, outType, Functor, PType>::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();
  auto x_dims = param.X->dims();
  size_t x_rank = x_dims.size();
  bool reduce_all = param.reduce_all;

  std::vector<int> dims;
  if (reduce_all || param.dim.size() == 0) {
    for (auto i = 0; i < x_rank; i++) {
      dims.push_back(i);
    }
  } else {
    dims = param.dim;
    for (size_t i = 0; i < dims.size(); ++i) {
      if (dims[i] < 0) {
        dims[i] = x_rank + dims[i];
      }
    }
  }
  std::stable_sort(dims.begin(), dims.end());
  std::vector<int> x_shape;
  for (size_t i = 0; i < x_dims.size(); i++) {
    x_shape.push_back(static_cast<int>(x_dims[i]));
  }

  float* quant_x_max = nullptr;
  float* quant_y_max = nullptr;
  Functor elt_func;

  if (!param.enable_int8) {
    int ret = elt_func(ctx.GetRawContext(),
                       param.X->template data<inType>(),
                       param.Out->template mutable_data<outType>(TARGET(kXPU)),
                       x_shape,
                       dims,
                       quant_x_max,
                       quant_y_max);
    CHECK_EQ(ret, 0);
  } else {
    quant_x_max = reinterpret_cast<float*>(quant_x_max_value_guard_->addr_);
    quant_y_max = reinterpret_cast<float*>(quant_out_max_value_guard_->addr_);
    int ret = elt_func(ctx.GetRawContext(),
                       param.X->template data<inType>(),
                       param.Out->template mutable_data<outType>(TARGET(kXPU)),
                       x_shape,
                       dims,
                       quant_x_max,
                       quant_y_max);
    CHECK_EQ(ret, 0);
  }

  return;
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

namespace xpu = paddle::lite::kernels::xpu;
using ReduceAll = xpu::
    ReduceCompute<bool, bool, xpu::ReduceAllFunctor<bool>, PRECISION(kFloat)>;
using ReduceAny = xpu::
    ReduceCompute<bool, bool, xpu::ReduceAnyFunctor<bool>, PRECISION(kFloat)>;
using ReduceMeanFloat32 = xpu::ReduceCompute<float,
                                             float,
                                             xpu::ReduceMeanFunctor<float>,
                                             PRECISION(kFloat)>;
using ReduceMeanFloat16 = xpu::ReduceCompute<float16,
                                             float16,
                                             xpu::ReduceMeanFunctor<float16>,
                                             PRECISION(kFP16)>;
using ReduceMeanInt8 = xpu::ReduceCompute<int8_t,
                                          int8_t,
                                          xpu::ReduceMeanFunctor<int8_t>,
                                          PRECISION(kInt8)>;
using ReduceSumFloat32 = xpu::ReduceCompute<float,
                                            float,
                                            xpu::ReduceSumFunctor<float>,
                                            PRECISION(kFloat)>;
using ReduceSumFloat16 = xpu::ReduceCompute<float16,
                                            float16,
                                            xpu::ReduceSumFunctor<float16>,
                                            PRECISION(kFP16)>;
using ReduceSumInt32Int64 =
    xpu::ReduceCompute<int32_t,
                       int64_t,
                       xpu::ReduceSumMixFunctor<int32_t, int64_t>,
                       PRECISION(kFloat)>;

using ReduceProdFloat32 = xpu::ReduceCompute<float,
                                             float,
                                             xpu::ReduceProdFunctor<float>,
                                             PRECISION(kFloat)>;
using ReduceProdFloat16 = xpu::ReduceCompute<float16,
                                             float16,
                                             xpu::ReduceProdFunctor<float16>,
                                             PRECISION(kFP16)>;
using ReduceMaxFloat32 = xpu::ReduceCompute<float,
                                            float,
                                            xpu::ReduceMaxFunctor<float>,
                                            PRECISION(kFloat)>;
using ReduceMaxFloat16 = xpu::ReduceCompute<float16,
                                            float16,
                                            xpu::ReduceMaxFunctor<float16>,
                                            PRECISION(kFP16)>;
using ReduceMaxInt64 = xpu::ReduceCompute<int64_t,
                                          int64_t,
                                          xpu::ReduceMaxFunctor<int64_t>,
                                          PRECISION(kFloat)>;

using ReduceMinFloat32 = xpu::ReduceCompute<float,
                                            float,
                                            xpu::ReduceMinFunctor<float>,
                                            PRECISION(kFloat)>;
using ReduceMinFloat16 = xpu::ReduceCompute<float16,
                                            float16,
                                            xpu::ReduceMinFunctor<float16>,
                                            PRECISION(kFP16)>;

REGISTER_LITE_KERNEL(reduce_all, kXPU, kFloat, kNCHW, ReduceAll, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kBool))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kBool))})
    .Finalize();

REGISTER_LITE_KERNEL(reduce_any, kXPU, kFloat, kNCHW, ReduceAny, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kBool))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kBool))})
    .Finalize();

REGISTER_LITE_KERNEL(reduce_mean, kXPU, kFloat, kNCHW, ReduceMeanFloat32, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();

REGISTER_LITE_KERNEL(reduce_mean,
                     kXPU,
                     kFP16,
                     kNCHW,
                     ReduceMeanFloat16,
                     DISABLE_XPU1_ReduceMeanFloat16)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .Finalize();

REGISTER_LITE_KERNEL(
    reduce_mean, kXPU, kInt8, kNCHW, ReduceMeanInt8, ReduceMeanInt8)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt8))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt8))})
    .Finalize();

REGISTER_LITE_KERNEL(reduce_sum, kXPU, kFloat, kNCHW, ReduceSumFloat32, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();

REGISTER_LITE_KERNEL(reduce_sum,
                     kXPU,
                     kFP16,
                     kNCHW,
                     ReduceSumFloat16,
                     DISABLE_XPU1_ReduceSumFloat16)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .Finalize();

REGISTER_LITE_KERNEL(reduce_sum,
                     kXPU,
                     kFloat,
                     kNCHW,
                     ReduceSumInt32Int64,
                     DISABLE_XPU1_ReduceSumInt32)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt64))})
    .Finalize();

REGISTER_LITE_KERNEL(reduce_prod, kXPU, kFloat, kNCHW, ReduceProdFloat32, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();

REGISTER_LITE_KERNEL(reduce_prod,
                     kXPU,
                     kFP16,
                     kNCHW,
                     ReduceProdFloat16,
                     DISABLE_XPU1_ReduceProdFloat16)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .Finalize();

REGISTER_LITE_KERNEL(reduce_max, kXPU, kFloat, kNCHW, ReduceMaxFloat32, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();

REGISTER_LITE_KERNEL(reduce_max,
                     kXPU,
                     kFP16,
                     kNCHW,
                     ReduceMaxFloat16,
                     DISABLE_XPU1_ReduceMaxFloat16)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .Finalize();

REGISTER_LITE_KERNEL(reduce_max,
                     kXPU,
                     kFloat,
                     kNCHW,
                     ReduceMaxInt64,
                     DISABLE_XPU1_ReduceMaxInt64)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt64))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt64))})
    .Finalize();

REGISTER_LITE_KERNEL(reduce_min, kXPU, kFloat, kNCHW, ReduceMinFloat32, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();

REGISTER_LITE_KERNEL(reduce_min,
                     kXPU,
                     kFP16,
                     kNCHW,
                     ReduceMinFloat16,
                     DISABLE_XPU1_ReduceMinFloat16)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .Finalize();
