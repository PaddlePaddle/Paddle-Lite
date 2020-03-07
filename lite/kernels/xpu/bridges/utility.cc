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

#include "lite/kernels/xpu/bridges/utility.h"
#include <utility>

namespace paddle {
namespace lite {
namespace subgraph {
namespace xpu {

bool HasInputArg(const OpInfo* op_info,
                 const Scope* scope,
                 const std::string& argname) {
  auto iarg_names = op_info->input_argnames();
  if (std::find(iarg_names.begin(), iarg_names.end(), argname) !=
      iarg_names.end()) {
    auto inputs = op_info->Input(argname);
    if (inputs.empty()) {
      return false;
    }
    auto var_name = inputs.front();
    auto var = scope->FindVar(var_name);
    return var != nullptr;
  } else {
    return false;
  }
}

xtcl::DataType CvtPrecisionType(PrecisionType in_type) {
  xtcl::DataType out_type = ::xtcl::Float(32);
  switch (in_type) {
    case PRECISION(kFloat):
      out_type = ::xtcl::Float(32);
      break;
    case PRECISION(kInt8):
      out_type = ::xtcl::Int(8);
      break;
    case PRECISION(kInt16):
      out_type = ::xtcl::Int(16);
      break;
    case PRECISION(kInt32):
      out_type = ::xtcl::Int(32);
      break;
    case PRECISION(kInt64):
      out_type = ::xtcl::Int(64);
      break;
    default:
      LOG(FATAL) << "[XPU] Can not convert precision type("
                 << PrecisionToStr(in_type) << ") from Lite to XPU";
      break;
  }
  return out_type;
}

DLDataType CvtDLDataType(PrecisionType in_type) {
  DLDataType out_type = {kDLFloat, 32, 1};
  switch (in_type) {
    case PRECISION(kFloat):
      out_type = {kDLFloat, 32, 1};
      break;
    case PRECISION(kInt8):
      out_type = {kDLInt, 8, 1};
      break;
    case PRECISION(kInt16):
      out_type = {kDLInt, 16, 1};
      break;
    case PRECISION(kInt32):
      out_type = {kDLInt, 32, 1};
      break;
    case PRECISION(kInt64):
      out_type = {kDLInt, 64, 1};
      break;
    default:
      LOG(FATAL) << "[XPU] Can not convert precision type("
                 << PrecisionToStr(in_type) << ") from Lite to XPU DLDataType";
      break;
  }
  return out_type;
}

DLDeviceType CvtDLDeviceType(TargetType in_type) {
  DLDeviceType out_type = kDLCPU;
  switch (in_type) {
    case TARGET(kX86):
      out_type = kDLCPU;
      break;
    case TARGET(kHost):
      out_type = kDLCPU;
      break;
    case TARGET(kCUDA):
      out_type = kDLGPU;
      break;
    case TARGET(kXPU):
      out_type = static_cast<DLDeviceType>(kDLXPU);
      break;
    default:
      LOG(FATAL) << "[XPU] Can not convert target type(" << TargetToStr(in_type)
                 << ") from Lite to XPU DLDeviceType";
      break;
  }
  return out_type;
}

std::shared_ptr<xtcl::xNDArray> CvtTensor(const Tensor& in_tensor,
                                          std::vector<int64_t> out_shape,
                                          DataLayoutType in_layout) {
  PrecisionType in_precision = in_tensor.precision();
  auto in_shape = in_tensor.dims().Vectorize();
  if (out_shape.empty()) {
    out_shape = in_shape;
  }
  auto out_tensor = std::make_shared<xtcl::xNDArray>(
      xtcl::xNDArray::Empty(out_shape,
                            CvtDLDataType(in_precision),
                            {CvtDLDeviceType(TARGET(kHost)), 0}));
  auto out_data =
      reinterpret_cast<uint8_t*>(out_tensor->ToDLPack()->dl_tensor.data);
  std::memcpy(out_data, in_tensor.raw_data(), in_tensor.memory_size());
  return out_tensor;
}

}  // namespace xpu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle
