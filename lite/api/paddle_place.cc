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

#include "lite/api/paddle_place.h"
#include "lite/utils/hash.h"
#include "lite/utils/log/cp_logging.h"
#include "lite/utils/replace_stl/stream.h"
#include "lite/utils/string.h"

namespace paddle {
namespace lite_api {

size_t Place::hash() const {
  std::hash<int> h;
  size_t hash = h(static_cast<int>(target));
  lite::CombineHash(static_cast<int64_t>(precision), &hash);
  lite::CombineHash(static_cast<int64_t>(layout), &hash);
  lite::CombineHash(static_cast<int64_t>(device), &hash);
  return hash;
}

bool operator<(const Place& a, const Place& b) {
  if (a.target != b.target) return a.target < b.target;
  if (a.precision != b.precision) return a.precision < b.precision;
  if (a.layout != b.layout) return a.layout < b.layout;
  if (a.device != b.device) return a.device < b.device;
  return false;
}

std::string Place::DebugString() const {
  STL::stringstream os;
  os << TargetToStr(target) << "/" << PrecisionToStr(precision) << "/"
     << DataLayoutToStr(layout);
  return os.str();
}

const std::string& ActivationTypeToStr(ActivationType act) {
  static const std::string act2string[] = {"unk",
                                           "Relu",
                                           "Relu6",
                                           "PRelu",
                                           "LeakyRelu",
                                           "Sigmoid",
                                           "Tanh",
                                           "Swish",
                                           "Exp",
                                           "Abs",
                                           "HardSwish",
                                           "Reciprocal",
                                           "ThresholdedRelu",
                                           "Elu",
                                           "HardSigmoid",
                                           "log"};
  auto x = static_cast<int>(act);
  CHECK_LT(x, static_cast<int>(ActivationType::NUM));
  return act2string[x];
}

const std::string& TargetToStr(TargetType target) {
  static const std::string target2string[] = {"unk",
                                              "host",
                                              "x86",
                                              "cuda",
                                              "arm",
                                              "opencl",
                                              "any",
                                              "fpga",
                                              "npu",
                                              "xpu",
                                              "bm",
                                              "mlu",
                                              "rknpu",
                                              "apu",
                                              "huawei_ascend_npu",
                                              "imagination_nna",
                                              "intel_fpga",
                                              "metal",
                                              "nnadapter"};
  auto x = static_cast<int>(target);

  CHECK_LT(x, static_cast<int>(TARGET(NUM)));
  return target2string[x];
}

const std::string& PrecisionToStr(PrecisionType precision) {
  static const std::string precision2string[] = {"unk",
                                                 "float",
                                                 "int8_t",
                                                 "int32_t",
                                                 "any",
                                                 "float16",
                                                 "bool",
                                                 "int64_t",
                                                 "int16_t",
                                                 "uint8_t",
                                                 "double"};
  auto x = static_cast<int>(precision);
  CHECK_LT(x, static_cast<int>(PRECISION(NUM)));
  return precision2string[x];
}

const std::string& DataLayoutToStr(DataLayoutType layout) {
  static const std::string datalayout2string[] = {"unk",
                                                  "NCHW",
                                                  "any",
                                                  "NHWC",
                                                  "ImageDefault",
                                                  "ImageFolder",
                                                  "ImageNW",
                                                  "MetalTexture2DArray",
                                                  "MetalTexture2D"};
  auto x = static_cast<int>(layout);
  CHECK_LT(x, static_cast<int>(DATALAYOUT(NUM)));
  return datalayout2string[x];
}

const std::string& TargetRepr(TargetType target) {
  static const std::string target2string[] = {"kUnk",
                                              "kHost",
                                              "kX86",
                                              "kCUDA",
                                              "kARM",
                                              "kOpenCL",
                                              "kAny",
                                              "kFPGA",
                                              "kNPU",
                                              "kXPU",
                                              "kBM",
                                              "kMLU",
                                              "kRKNPU",
                                              "kAPU",
                                              "kHuaweiAscendNPU",
                                              "kImaginationNNA",
                                              "kIntelFPGA",
                                              "kMetal",
                                              "kNNAdapter"};
  auto x = static_cast<int>(target);
  CHECK_LT(x, static_cast<int>(TARGET(NUM)));
  return target2string[x];
}

const std::string& PrecisionRepr(PrecisionType precision) {
  static const std::string precision2string[] = {"kUnk",
                                                 "kFloat",
                                                 "kInt8",
                                                 "kInt32",
                                                 "kAny",
                                                 "kFP16",
                                                 "kBool",
                                                 "kInt64",
                                                 "kInt16"};
  auto x = static_cast<int>(precision);
  CHECK_LT(x, static_cast<int>(PRECISION(NUM)));
  return precision2string[x];
}

const std::string& DataLayoutRepr(DataLayoutType layout) {
  static const std::string datalayout2string[] = {"kUnk",
                                                  "kNCHW",
                                                  "kAny",
                                                  "kNHWC",
                                                  "kImageDefault",
                                                  "kImageFolder",
                                                  "kImageNW",
                                                  "kMetalTexture2DArray",
                                                  "kMetalTexture2D"};
  auto x = static_cast<int>(layout);
  CHECK_LT(x, static_cast<int>(DATALAYOUT(NUM)));
  return datalayout2string[x];
}

const std::string& CLTuneModeToStr(CLTuneMode mode) {
  static const std::string cl_tune_mode[] = {
      "CL_TUNE_NONE", "CL_TUNE_RAPID", "CL_TUNE_NORMAL", "CL_TUNE_EXHAUSTIVE"};
  auto x = static_cast<int>(mode);
  return cl_tune_mode[x];
}

const std::string& CLPrecisionTypeToStr(CLPrecisionType type) {
  static const std::string cl_precision_type[] = {
      "CL_PRECISION_AUTO", "CL_PRECISION_FP32", "CL_PRECISION_FP16"};
  auto x = static_cast<int>(type);
  return cl_precision_type[x];
}

std::set<TargetType> ExpandValidTargets(TargetType target) {
  static const std::set<TargetType> valid_set({TARGET(kHost),
                                               TARGET(kX86),
                                               TARGET(kCUDA),
                                               TARGET(kARM),
                                               TARGET(kOpenCL),
                                               TARGET(kNPU),
                                               TARGET(kXPU),
                                               TARGET(kBM),
                                               TARGET(kMLU),
                                               TARGET(kAPU),
                                               TARGET(kRKNPU),
                                               TARGET(kFPGA),
                                               TARGET(kHuaweiAscendNPU),
                                               TARGET(kImaginationNNA),
                                               TARGET(kIntelFPGA),
                                               TARGET(kMetal),
                                               TARGET(kNNAdapter)});
  if (target == TARGET(kAny)) {
    return valid_set;
  }
  return std::set<TargetType>({target});
}

std::set<PrecisionType> ExpandValidPrecisions(PrecisionType precision) {
  static const std::set<PrecisionType> valid_set(
      {PRECISION(kFloat), PRECISION(kInt8), PRECISION(kFP16), PRECISION(kAny)});
  if (precision == PRECISION(kAny)) {
    return valid_set;
  }
  return std::set<PrecisionType>({precision});
}

std::set<DataLayoutType> ExpandValidLayouts(DataLayoutType layout) {
  static const std::set<DataLayoutType> valid_set(
      {DATALAYOUT(kNCHW),
       DATALAYOUT(kAny),
       DATALAYOUT(kNHWC),
       DATALAYOUT(kImageDefault),
       DATALAYOUT(kImageFolder),
       DATALAYOUT(kImageNW),
       DATALAYOUT(kMetalTexture2DArray),
       DATALAYOUT(kMetalTexture2D)});
  if (layout == DATALAYOUT(kAny)) {
    return valid_set;
  }
  return std::set<DataLayoutType>({layout});
}

}  // namespace lite_api
}  // namespace paddle
