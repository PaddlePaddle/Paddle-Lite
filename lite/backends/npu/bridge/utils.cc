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

#include "lite/backends/npu/bridge/utils.h"
#include <memory>
#include <mutex>  // NOLINT
#include <string>
#include <unordered_map>
#include "ai_ddk_lib/include/graph/op/all_ops.h"  // for ge::op::Data
#include "ai_ddk_lib/include/graph/tensor.h"      // for ge::TensorUtils
#include "lite/core/op_lite.h"

namespace paddle {
namespace lite {
namespace npu {
namespace bridge {

std::string UniqueName(const std::string& prefix) {
  static std::mutex counter_mtx;
  static std::unordered_map<std::string, int> counter_map;
  std::unique_lock<std::mutex> counter_lck(counter_mtx);
  int counter = 1;
  auto it = counter_map.find(prefix);
  if (it == counter_map.end()) {
    counter_map[prefix] = counter;
  } else {
    counter = ++(it->second);
  }
  return prefix + "_" + std::to_string(counter);
}

ge::DataType PrecisionConverter(PrecisionType itype) {
  ge::DataType otype = ge::DT_FLOAT;
  switch (itype) {
    case PRECISION(kFloat):
      otype = ge::DT_FLOAT;
      break;
    case PRECISION(kInt8):
      otype = ge::DT_INT8;
      break;
    case PRECISION(kInt32):
      otype = ge::DT_INT32;
      break;
    default:
      LOG(FATAL) << "Can not convert precision type(" << PrecisionToStr(itype)
                 << ") from Lite to NPU";
      break;
  }
  return otype;
}

ge::Format DataLayoutConverter(DataLayoutType itype) {
  ge::Format otype = ge::FORMAT_NCHW;
  switch (itype) {
    case DATALAYOUT(kNCHW):
      otype = ge::FORMAT_NCHW;
      break;
    // TODO(hong19860320) support more data layout type
    default:
      LOG(FATAL) << "Can not convert data layout type("
                 << DataLayoutToStr(itype) << ") from Lite to NPU";
      break;
  }
  return otype;
}

ge::TensorPtr CvtFromLiteTensor(lite::Tensor* in_tensor,
                                std::vector<int64_t> out_shape,
                                PrecisionType in_ptype,
                                DataLayoutType in_ltype) {
  uint8_t* in_data = nullptr;
  auto in_size = in_tensor->dims().production();
  auto in_shape = in_tensor->dims().Vectorize();
  if (out_shape.empty()) {
    out_shape = in_shape;
  }
  int in_bytes;
  if (in_ptype == PRECISION(kFloat)) {
    in_data = reinterpret_cast<uint8_t*>(in_tensor->mutable_data<float>());
    in_bytes = in_size * sizeof(float);
  } else if (in_ptype == PRECISION(kInt32)) {
    in_data = reinterpret_cast<uint8_t*>(in_tensor->mutable_data<int32_t>());
    in_bytes = in_size * sizeof(int32_t);
  } else if (in_ptype == PRECISION(kInt8)) {
    in_data = reinterpret_cast<uint8_t*>(in_tensor->mutable_data<int8_t>());
    in_bytes = in_size * sizeof(int8_t);
  } else {
    LOG(FATAL) << "Unknow precision type " << PrecisionToStr(in_ptype);
  }
  ge::DataType out_ptype = PrecisionConverter(in_ptype);
  ge::Format out_ltype = DataLayoutConverter(in_ltype);

  ge::TensorDesc out_desc(ge::Shape(out_shape), out_ltype, out_ptype);
  CHECK_EQ(out_ltype, ge::FORMAT_NCHW);

  auto out_size = out_desc.GetShape().GetShapeSize();
  CHECK_EQ(out_size, in_size);

  ge::TensorPtr out_tensor = std::make_shared<ge::Tensor>();
  out_tensor->SetTensorDesc(out_desc);
  out_tensor->SetData(in_data, in_bytes);
  return out_tensor;
}

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

}  // namespace bridge
}  // namespace npu
}  // namespace lite
}  // namespace paddle
