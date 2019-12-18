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
    case PRECISION(kInt32):
      out_type = ::xtcl::Int(32);
      break;
    default:
      LOG(FATAL) << "[XPU] Can not convert precision type("
                 << PrecisionToStr(in_type) << ") from Lite to XPU";
      break;
  }
  return out_type;
}

DLDataType CvtDataType(PrecisionType in_type) {
  DLDataType out_type = {kDLFloat, 32, 1};
  switch (in_type) {
    case PRECISION(kFloat):
      out_type = {kDLFloat, 32, 1};
      break;
    case PRECISION(kInt8):
      out_type = {kDLInt, 8, 1};
      break;
    case PRECISION(kInt32):
      out_type = {kDLInt, 32, 1};
      break;
    default:
      LOG(FATAL) << "[XPU] Can not convert data type("
                 << PrecisionToStr(in_type) << ") from Lite to XPU";
      break;
  }
  return out_type;
}

xtcl::Array<xtcl::xIndexExpr> CvtShape(const std::vector<int>& in_shape) {
  xtcl::Array<xtcl::xIndexExpr> out_shape;
  for (auto dim : in_shape) {
    out_shape.push_back(dim);
  }
  return out_shape;
}

xtcl::Array<xtcl::xIndexExpr> CvtShape(const std::vector<int64_t>& in_shape) {
  return CvtShape(std::vector<int>(in_shape.begin(), in_shape.end()));
}

xtcl::Array<xtcl::xIndexExpr> CvtShape(const DDim& in_dims) {
  return CvtShape(in_dims.Vectorize());
}

std::shared_ptr<xtcl::xNDArray> CvtTensor(const Tensor& in_tensor,
                                          std::vector<int64_t> out_shape,
                                          PrecisionType in_ptype,
                                          DataLayoutType in_ltype) {
  const uint8_t* in_data = nullptr;
  auto in_size = in_tensor.dims().production();
  auto in_shape = in_tensor.dims().Vectorize();
  if (out_shape.empty()) {
    out_shape = in_shape;
  }
  int in_bytes;
  if (in_ptype == PRECISION(kFloat)) {
    in_data = reinterpret_cast<const uint8_t*>(in_tensor.data<float>());
    in_bytes = in_size * sizeof(float);
  } else if (in_ptype == PRECISION(kInt32)) {
    in_data = reinterpret_cast<const uint8_t*>(in_tensor.data<int32_t>());
    in_bytes = in_size * sizeof(int32_t);
  } else if (in_ptype == PRECISION(kInt8)) {
    in_data = reinterpret_cast<const uint8_t*>(in_tensor.data<int8_t>());
    in_bytes = in_size * sizeof(int8_t);
  } else {
    LOG(FATAL) << "[XPU] Unknow precision type " << PrecisionToStr(in_ptype);
  }
  auto out_tensor = std::make_shared<xtcl::xNDArray>(
      xtcl::xNDArray::Empty(out_shape, CvtDataType(in_ptype), {kDLCPU, 0}));
  auto out_data =
      reinterpret_cast<uint8_t*>(out_tensor->ToDLPack()->dl_tensor.data);
  std::memcpy(out_data, in_data, in_bytes);
  return out_tensor;
}

}  // namespace xpu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle
