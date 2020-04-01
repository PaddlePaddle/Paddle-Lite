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

#include "lite/core/op_registry.h"
#include <list>
#include <set>

namespace paddle {
namespace lite {

const std::map<std::string, std::string> &GetOp2PathDict() {
  return OpKernelInfoCollector::Global().GetOp2PathDict();
}

std::list<std::unique_ptr<KernelBase>> KernelRegistry::Create(
    const std::string &op_type,
    TargetType target,
    PrecisionType precision,
    DataLayoutType layout) {
  Place place{target, precision, layout};
  VLOG(5) << "creating " << op_type << " kernel for " << place.DebugString();
#define CREATE_KERNEL1(target__, precision__)                                \
  switch (layout) {                                                          \
    case DATALAYOUT(kNCHW):                                                  \
      return Create<TARGET(target__),                                        \
                    PRECISION(precision__),                                  \
                    DATALAYOUT(kNCHW)>(op_type);                             \
    case DATALAYOUT(kAny):                                                   \
      return Create<TARGET(target__),                                        \
                    PRECISION(precision__),                                  \
                    DATALAYOUT(kAny)>(op_type);                              \
    case DATALAYOUT(kNHWC):                                                  \
      return Create<TARGET(target__),                                        \
                    PRECISION(precision__),                                  \
                    DATALAYOUT(kNHWC)>(op_type);                             \
    case DATALAYOUT(kImageDefault):                                          \
      return Create<TARGET(target__),                                        \
                    PRECISION(precision__),                                  \
                    DATALAYOUT(kImageDefault)>(op_type);                     \
    case DATALAYOUT(kImageFolder):                                           \
      return Create<TARGET(target__),                                        \
                    PRECISION(precision__),                                  \
                    DATALAYOUT(kImageFolder)>(op_type);                      \
    case DATALAYOUT(kImageNW):                                               \
      return Create<TARGET(target__),                                        \
                    PRECISION(precision__),                                  \
                    DATALAYOUT(kImageNW)>(op_type);                          \
    default:                                                                 \
      LOG(FATAL) << "unsupported kernel layout " << DataLayoutToStr(layout); \
  }

#define CREATE_KERNEL(target__)                         \
  switch (precision) {                                  \
    case PRECISION(kFloat):                             \
      CREATE_KERNEL1(target__, kFloat);                 \
    case PRECISION(kInt8):                              \
      CREATE_KERNEL1(target__, kInt8);                  \
    case PRECISION(kFP16):                              \
      CREATE_KERNEL1(target__, kFP16);                  \
    case PRECISION(kAny):                               \
      CREATE_KERNEL1(target__, kAny);                   \
    case PRECISION(kInt32):                             \
      CREATE_KERNEL1(target__, kInt32);                 \
    case PRECISION(kInt64):                             \
      CREATE_KERNEL1(target__, kInt64);                 \
    default:                                            \
      CHECK(false) << "not supported kernel precision " \
                   << PrecisionToStr(precision);        \
  }

  switch (target) {
    case TARGET(kHost): {
      CREATE_KERNEL(kHost);
    } break;
    case TARGET(kX86): {
      CREATE_KERNEL(kX86);
    } break;
    case TARGET(kCUDA): {
      CREATE_KERNEL(kCUDA);
    } break;
    case TARGET(kARM): {
      CREATE_KERNEL(kARM);
    } break;
    case TARGET(kOpenCL): {
      CREATE_KERNEL(kOpenCL);
    } break;
    case TARGET(kNPU): {
      CREATE_KERNEL(kNPU);
    } break;
    case TARGET(kXPU): {
      CREATE_KERNEL(kXPU);
    } break;
    case TARGET(kFPGA): {
      CREATE_KERNEL(kFPGA);
    } break;
    case TARGET(kBM): {
      CREATE_KERNEL(kBM);
    } break;
    case TARGET(kMLU): {
      CREATE_KERNEL(kMLU);
    } break;
    default:
      CHECK(false) << "not supported kernel target " << TargetToStr(target);
  }

#undef CREATE_KERNEL
  return std::list<std::unique_ptr<KernelBase>>();
}

KernelRegistry::KernelRegistry()
    : registries_(static_cast<int>(TARGET(NUM)) *
                  static_cast<int>(PRECISION(NUM)) *
                  static_cast<int>(DATALAYOUT(NUM))) {
#define INIT_FOR(target__, precision__, layout__)                      \
  registries_[KernelRegistry::GetKernelOffset<TARGET(target__),        \
                                              PRECISION(precision__),  \
                                              DATALAYOUT(layout__)>()] \
      .set<KernelRegistryForTarget<TARGET(target__),                   \
                                   PRECISION(precision__),             \
                                   DATALAYOUT(layout__)> *>(           \
          &KernelRegistryForTarget<TARGET(target__),                   \
                                   PRECISION(precision__),             \
                                   DATALAYOUT(layout__)>::Global());
  // Currently, just register 2 kernel targets.
  INIT_FOR(kCUDA, kFloat, kNCHW);
  INIT_FOR(kCUDA, kFloat, kNHWC);
  INIT_FOR(kCUDA, kInt8, kNCHW);
  INIT_FOR(kCUDA, kAny, kNCHW);
  INIT_FOR(kCUDA, kAny, kAny);
  INIT_FOR(kCUDA, kInt8, kNHWC);
  INIT_FOR(kCUDA, kInt64, kNCHW);
  INIT_FOR(kCUDA, kInt64, kNHWC);

  INIT_FOR(kMLU, kFloat, kNHWC);
  INIT_FOR(kMLU, kFloat, kNCHW);
  INIT_FOR(kMLU, kFP16, kNHWC);
  INIT_FOR(kMLU, kFP16, kNCHW);
  INIT_FOR(kMLU, kInt8, kNHWC);
  INIT_FOR(kMLU, kInt8, kNCHW);
  INIT_FOR(kMLU, kInt16, kNHWC);
  INIT_FOR(kMLU, kInt16, kNCHW);

  INIT_FOR(kHost, kFloat, kNCHW);
  INIT_FOR(kHost, kAny, kNCHW);
  INIT_FOR(kHost, kFloat, kNHWC);
  INIT_FOR(kHost, kFloat, kAny);
  INIT_FOR(kHost, kAny, kNHWC);
  INIT_FOR(kHost, kAny, kAny);
  INIT_FOR(kHost, kAny, kNHWC);
  INIT_FOR(kHost, kAny, kAny);

  INIT_FOR(kX86, kFloat, kNCHW);
  INIT_FOR(kX86, kAny, kNCHW);
  INIT_FOR(kX86, kAny, kAny);
  INIT_FOR(kX86, kInt64, kNCHW);

  INIT_FOR(kARM, kFloat, kNCHW);
  INIT_FOR(kARM, kFloat, kNHWC);
  INIT_FOR(kARM, kInt8, kNCHW);
  INIT_FOR(kARM, kInt8, kNHWC);
  INIT_FOR(kARM, kAny, kNCHW);
  INIT_FOR(kARM, kAny, kAny);
  INIT_FOR(kARM, kInt32, kNCHW);
  INIT_FOR(kARM, kInt64, kNCHW);

  INIT_FOR(kOpenCL, kFloat, kNCHW);
  INIT_FOR(kOpenCL, kFloat, kNHWC);
  INIT_FOR(kOpenCL, kAny, kNCHW);
  INIT_FOR(kOpenCL, kAny, kNHWC);
  INIT_FOR(kOpenCL, kFloat, kAny);
  INIT_FOR(kOpenCL, kInt8, kNCHW);
  INIT_FOR(kOpenCL, kAny, kAny);
  INIT_FOR(kOpenCL, kFP16, kNCHW);
  INIT_FOR(kOpenCL, kFP16, kNHWC);
  INIT_FOR(kOpenCL, kFP16, kImageDefault);
  INIT_FOR(kOpenCL, kFP16, kImageFolder);
  INIT_FOR(kOpenCL, kFP16, kImageNW);
  INIT_FOR(kOpenCL, kFloat, kImageDefault);
  INIT_FOR(kOpenCL, kFloat, kImageFolder);
  INIT_FOR(kOpenCL, kFloat, kImageNW);
  INIT_FOR(kOpenCL, kAny, kImageDefault);
  INIT_FOR(kOpenCL, kAny, kImageFolder);
  INIT_FOR(kOpenCL, kAny, kImageNW);

  INIT_FOR(kNPU, kFloat, kNCHW);
  INIT_FOR(kNPU, kFloat, kNHWC);
  INIT_FOR(kNPU, kInt8, kNCHW);
  INIT_FOR(kNPU, kInt8, kNHWC);
  INIT_FOR(kNPU, kAny, kNCHW);
  INIT_FOR(kNPU, kAny, kNHWC);
  INIT_FOR(kNPU, kAny, kAny);

  INIT_FOR(kXPU, kFloat, kNCHW);
  INIT_FOR(kXPU, kInt8, kNCHW);
  INIT_FOR(kXPU, kAny, kNCHW);
  INIT_FOR(kXPU, kAny, kAny);

  INIT_FOR(kFPGA, kFP16, kNHWC);
  INIT_FOR(kFPGA, kFP16, kAny);
  INIT_FOR(kFPGA, kFloat, kNHWC);
  INIT_FOR(kFPGA, kAny, kNHWC);
  INIT_FOR(kFPGA, kAny, kAny);

  INIT_FOR(kBM, kFloat, kNCHW);
  INIT_FOR(kBM, kInt8, kNCHW);
  INIT_FOR(kBM, kAny, kNCHW);
  INIT_FOR(kBM, kAny, kAny);
#undef INIT_FOR
}

KernelRegistry &KernelRegistry::Global() {
  static auto *x = new KernelRegistry;
  return *x;
}

}  // namespace lite
}  // namespace paddle
