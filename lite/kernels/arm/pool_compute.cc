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

#include "lite/kernels/arm/pool_compute.h"
#include <string>
#include <vector>
#include "lite/backends/arm/math/funcs.h"
#include "lite/core/op_registry.h"
#include "lite/core/type_system.h"
#ifdef ENABLE_ARM_FP16
#include "lite/backends/arm/math/fp16/funcs_fp16.h"
#endif
#if defined(__aarch64__) && defined(LITE_WITH_ARM8_SVE2)
#include "lite/backends/arm/math/sve/pooling_sve.h"
#endif

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

#define POOL_IN_PARAM                                                        \
  din, dout, out_dims[0], out_dims[1], out_dims[2], out_dims[3], in_dims[1], \
      in_dims[2], in_dims[3]
template <>
void PoolCompute<PRECISION(kFloat), PRECISION(kFloat)>::Run() {
  auto& param = Param<operators::PoolParam>();
  auto& in_dims = param.x->dims();
  auto& out_dims = param.output->dims();
  auto& ctx = this->ctx_->As<ARMContext>();

  const float* din = param.x->data<float>();
  float* dout = param.output->mutable_data<float>();

  std::vector<int>& ksize = param.ksize;
  std::vector<int>& strides = param.strides;
  std::vector<int>& paddings = *param.paddings;

  std::string& pooling_type = param.pooling_type;
  bool exclusive = param.exclusive;
  bool adaptive = param.adaptive;
  bool ceil_mode = param.ceil_mode;
  bool use_quantizer = param.use_quantizer;
  std::string& data_format = param.data_format;

  bool pads_less =
      (paddings[0] == paddings[2]) && (paddings[1] < 2) && (paddings[3] < 2);

  bool pads_equal = (paddings[0] == paddings[2]) &&
                    (paddings[0] == paddings[1]) &&
                    (paddings[2] == paddings[3]);
  bool kps_equal =
      (ksize[0] == ksize[1]) && (strides[0] == strides[1]) && pads_less;
  bool global_pooling = (paddings[0] == 0) && (ksize[0] == in_dims[2]) &&
                        (ksize[1] == in_dims[3]) && kps_equal && pads_equal;
  bool win_ksize = (in_dims[2] > ksize[0]) && (in_dims[3] > ksize[1]);
  global_pooling = param.global_pooling || global_pooling;
  kps_equal = kps_equal && win_ksize;
  auto x_dims = param.x->dims();
  auto w_in = x_dims[x_dims.size() - 1];

  if (global_pooling) {
    for (size_t i = 0; i < ksize.size(); ++i) {
      paddings[2 * i] = 0;
      paddings[2 * i + 1] = 0;
      ksize[i] = static_cast<int>(in_dims[i + 2]);
    }
    if (pooling_type == "max") {
      lite::arm::math::pooling_global_max(POOL_IN_PARAM);
      return;
    } else if (pooling_type == "avg") {
#if defined(__aarch64__) && defined(LITE_WITH_ARM8_SVE2)
      if (ctx.has_sve2()) {
        lite::arm::math::pooling_global_avg_sve(POOL_IN_PARAM);
        return;
      }
#endif
      lite::arm::math::pooling_global_avg(POOL_IN_PARAM);
      return;
    }
  } else {
    if (w_in > 8 && ksize[0] == 1 && strides[0] == 2 && paddings[0] == 0 &&
        kps_equal) {
      // auto& ctx = this->ctx_->template As<ARMContext>();
      if (pooling_type == "max") {
        lite::arm::math::pooling1x1s2p0_max(
            POOL_IN_PARAM, paddings[1], paddings[3]);
        return;
      }
    } else if (w_in > 8 && ksize[0] == 2 && strides[0] == 2 &&
               paddings[0] == 0 && kps_equal) {
      if (pooling_type == "max") {
        lite::arm::math::pooling2x2s2p0_max(
            POOL_IN_PARAM, paddings[1], paddings[3]);
        return;
      } else if (pooling_type == "avg") {
        lite::arm::math::pooling2x2s2p0_avg(
            POOL_IN_PARAM, exclusive, paddings[1], paddings[3]);
        return;
      }
    } else if (w_in > 8 && ksize[0] == 2 && strides[0] == 2 &&
               paddings[0] == 1 && kps_equal) {
      if (pooling_type == "max") {
        lite::arm::math::pooling2x2s2p1_max(
            POOL_IN_PARAM, paddings[1], paddings[3]);
        return;
      } else if (pooling_type == "avg") {
        lite::arm::math::pooling2x2s2p1_avg(
            POOL_IN_PARAM, exclusive, paddings[1], paddings[3]);
        return;
      }
    } else if (ksize[0] == 3 && strides[0] == 1 && paddings[0] == 1 &&
               pads_equal && kps_equal) {
      if (pooling_type == "max") {
        lite::arm::math::pooling3x3s1p1_max(
            POOL_IN_PARAM, paddings[1], paddings[3]);
        return;
      } else if (pooling_type == "avg") {
        lite::arm::math::pooling3x3s1p1_avg(
            POOL_IN_PARAM, exclusive, paddings[1], paddings[3]);
        return;
      }
    } else if (ksize[0] == 3 && strides[0] == 1 && paddings[0] == 0 &&
               pads_equal && kps_equal) {
      if (pooling_type == "max") {
        lite::arm::math::pooling3x3s1p0_max(
            POOL_IN_PARAM, paddings[1], paddings[3]);
        return;
      } else if (pooling_type == "avg") {
        lite::arm::math::pooling3x3s1p0_avg(
            POOL_IN_PARAM, exclusive, paddings[1], paddings[3]);
        return;
      }
    } else if (ksize[0] == 3 && strides[0] == 2 && paddings[0] == 0 &&
               pads_equal && kps_equal) {
      if (pooling_type == "max") {
        lite::arm::math::pooling3x3s2p0_max(
            POOL_IN_PARAM, paddings[1], paddings[3]);
        return;
      } else if (pooling_type == "avg") {
        lite::arm::math::pooling3x3s2p0_avg(
            POOL_IN_PARAM, exclusive, paddings[1], paddings[3]);
        return;
      }
    } else if (ksize[0] == 3 && strides[0] == 2 && paddings[0] == 1 &&
               pads_equal && kps_equal) {
      if (pooling_type == "max") {
        lite::arm::math::pooling3x3s2p1_max(
            POOL_IN_PARAM, paddings[1], paddings[3]);
        return;
      } else if (pooling_type == "avg") {
        lite::arm::math::pooling3x3s2p1_avg(
            POOL_IN_PARAM, exclusive, paddings[1], paddings[3]);
        return;
      }
    }
  }

  lite::arm::math::pooling_basic(POOL_IN_PARAM,
                                 ksize,
                                 strides,
                                 paddings,
                                 global_pooling,
                                 exclusive,
                                 adaptive,
                                 ceil_mode,
                                 use_quantizer,
                                 pooling_type);
}
#ifdef ENABLE_ARM_FP16
template <>
void PoolCompute<PRECISION(kFP16), PRECISION(kFP16)>::Run() {
  auto& param = Param<operators::PoolParam>();
  auto& in_dims = param.x->dims();
  auto& out_dims = param.output->dims();

  const float16_t* din = param.x->data<float16_t>();
  float16_t* dout = param.output->mutable_data<float16_t>();
  auto& ctx = this->ctx_->As<ARMContext>();

  std::vector<int>& ksize = param.ksize;
  std::vector<int>& strides = param.strides;
  std::vector<int>& paddings = *param.paddings;

  std::string& pooling_type = param.pooling_type;
  bool exclusive = param.exclusive;
  bool adaptive = param.adaptive;
  bool ceil_mode = param.ceil_mode;
  bool use_quantizer = param.use_quantizer;
  std::string& data_format = param.data_format;

  bool pads_less =
      (paddings[0] == paddings[2]) && (paddings[1] < 2) && (paddings[3] < 2);

  bool pads_equal = (paddings[0] == paddings[2]) &&
                    (paddings[0] == paddings[1]) &&
                    (paddings[2] == paddings[3]);
  bool kps_equal =
      (ksize[0] == ksize[1]) && (strides[0] == strides[1]) && pads_less;
  bool global_pooling = (paddings[0] == 0) && (ksize[0] == in_dims[2]) &&
                        (ksize[1] == in_dims[3]) && kps_equal && pads_equal;
  bool win_ksize = (in_dims[2] > ksize[0]) && (in_dims[3] > ksize[1]);
  global_pooling = param.global_pooling || global_pooling;
  kps_equal = kps_equal && win_ksize;
  auto x_dims = param.x->dims();
  auto w_in = x_dims[x_dims.size() - 1];

  if (global_pooling) {
    for (size_t i = 0; i < ksize.size(); ++i) {
      paddings[2 * i] = 0;
      paddings[2 * i + 1] = 0;
      ksize[i] = static_cast<int>(in_dims[i + 2]);
    }
    if (pooling_type == "max") {
      lite::arm::math::fp16::pooling_global_max_fp16(POOL_IN_PARAM);
      return;
    } else if (pooling_type == "avg") {
#if defined(__aarch64__) && defined(LITE_WITH_ARM8_SVE2)
      if (ctx.has_sve2()) {
        lite::arm::math::pooling_global_avg_fp16_sve(POOL_IN_PARAM);
        return;
      }
#endif
      lite::arm::math::fp16::pooling_global_avg_fp16(POOL_IN_PARAM);
      return;
    }
  } else if (ksize[0] == 3 && strides[0] == 2 && paddings[0] == 0 &&
             pads_equal && kps_equal) {
    if (pooling_type == "max") {
      lite::arm::math::fp16::pooling3x3s2p0_max_fp16(
          POOL_IN_PARAM, paddings[1], paddings[3]);
      return;
    } else if (pooling_type == "avg") {
      lite::arm::math::fp16::pooling3x3s2p0_avg_fp16(
          POOL_IN_PARAM, exclusive, paddings[1], paddings[3]);
      return;
    }
  } else if (ksize[0] == 3 && strides[0] == 2 && paddings[0] == 1 &&
             pads_equal && kps_equal) {
    if (pooling_type == "max") {
      lite::arm::math::fp16::pooling3x3s2p1_max_fp16(
          POOL_IN_PARAM, paddings[1], paddings[3]);
      return;
    } else if (pooling_type == "avg") {
      lite::arm::math::fp16::pooling3x3s2p1_avg_fp16(
          POOL_IN_PARAM, exclusive, paddings[1], paddings[3]);
      return;
    }
  } else if (ksize[0] == 3 && strides[0] == 1 && paddings[0] == 0 &&
             pads_equal && kps_equal) {
    if (pooling_type == "max") {
      lite::arm::math::fp16::pooling3x3s1p0_max_fp16(
          POOL_IN_PARAM, paddings[1], paddings[3]);
      return;
    } else if (pooling_type == "avg") {
      lite::arm::math::fp16::pooling3x3s1p0_avg_fp16(
          POOL_IN_PARAM, exclusive, paddings[1], paddings[3]);
      return;
    }
  } else if (ksize[0] == 3 && strides[0] == 1 && paddings[0] == 1 &&
             pads_equal && kps_equal) {
    if (pooling_type == "max") {
      lite::arm::math::fp16::pooling3x3s1p1_max_fp16(
          POOL_IN_PARAM, paddings[1], paddings[3]);
      return;
    } else if (pooling_type == "avg") {
      lite::arm::math::fp16::pooling3x3s1p1_avg_fp16(
          POOL_IN_PARAM, exclusive, paddings[1], paddings[3]);
      return;
    }
  }
  lite::arm::math::fp16::pooling_basic_fp16(POOL_IN_PARAM,
                                            ksize,
                                            strides,
                                            paddings,
                                            global_pooling,
                                            exclusive,
                                            adaptive,
                                            ceil_mode,
                                            use_quantizer,
                                            pooling_type);
}
#endif
#undef POOL_IN_PARAM
}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
#ifdef ENABLE_ARM_FP16
typedef paddle::lite::kernels::arm::PoolCompute<PRECISION(kFP16),
                                                PRECISION(kFP16)>
    PoolFp16;
REGISTER_LITE_KERNEL(pool2d, kARM, kFP16, kNCHW, PoolFp16, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFP16))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFP16))})
    .BindPaddleOpVersion("pool2d", 1)
    .Finalize();
#endif  // ENABLE_ARM_FP16

typedef paddle::lite::kernels::arm::PoolCompute<PRECISION(kFloat),
                                                PRECISION(kFloat)>
    PoolFp32;
REGISTER_LITE_KERNEL(pool2d, kARM, kFloat, kNCHW, PoolFp32, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindPaddleOpVersion("pool2d", 1)
    .Finalize();
