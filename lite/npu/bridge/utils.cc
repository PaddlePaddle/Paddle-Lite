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

#include "lite/npu/bridge/utils.h"
#include <memory>
#include <mutex>  // NOLINT
#include <string>
#include <unordered_map>
#include "ai_ddk_lib/include/graph/op/all_ops.h"  // for ge::op::Data
#include "ai_ddk_lib/include/graph/tensor.h"      // for ge::TensorUtils

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

  // if (out_ltype == ge::FORMAT_NCHW) {
  //   if (dims.size() == 1) {
  //     dims.push_back(dims[0]);
  //     dims[0] = 1;
  //     dims.push_back(1);
  //     dims.push_back(1);
  //   } else if (dims.size() < 4) {
  //     for (size_t i = 4 - dims.size(); i > 0; --i) {
  //       dims.push_back(1);
  //     }
  //   }
  //   CHECK_EQ(dims.size(), 4);
  // }

  ge::TensorDesc out_desc(ge::Shape(out_shape), out_ltype, out_ptype);
  CHECK_EQ(out_ltype, ge::FORMAT_NCHW);

  auto out_size = out_desc.GetShape().GetShapeSize();
  CHECK_EQ(out_size, in_size);

  ge::TensorPtr out_tensor = std::make_shared<ge::Tensor>();
  out_tensor->SetTensorDesc(out_desc);
  out_tensor->SetData(in_data, in_bytes);
  return out_tensor;
}

std::shared_ptr<ge::Operator> CvtNode(lite::mir::Node* var_node,
                                      const Scope* scope) {
  CHECK(var_node->IsArg());
  const auto& arg = var_node->AsArg();
  VLOG(4) << "Convert var node " << arg.name;

  auto* var = scope->FindVar(arg.name);
  CHECK(var);
  auto* tensor = var->GetMutable<lite::Tensor>();
  CHECK(tensor);
  auto dims = tensor->dims();
  if (arg.is_weight) {
    auto wgt = std::make_shared<ge::op::Const>(arg.name);
    LOG(INFO) << "in convert const:" << arg.name;
    LOG(INFO) << dims;
    wgt->set_attr_value(CvtFromLiteTensor(tensor));

    auto odesc = wgt->GetOutputDesc(0);
    LOG(INFO) << "const ----";
    for (auto i : odesc.GetShape().GetDims()) {
      LOG(INFO) << ";;;;;;;;;------: " << i;
    }
    return wgt;
  } else {
    CHECK_EQ(dims.size(), 4);
    LOG(INFO) << "in convert data:" << arg.name;
    LOG(INFO) << dims;
    // TODO(TJ): support more types and dims size
    ge::TensorDesc desc(ge::Shape(dims.Vectorize()),
                        ge::Format::FORMAT_NCHW,
                        ge::DataType::DT_FLOAT);

    //   auto size = desc.GetShape().GetShapeSize();
    //  ge::TensorUtils::SetSize(desc, size*sizeof(float));
    //  ge::TensorUtils::SetRealDimCnt(desc, 4);
    auto data = std::make_shared<ge::op::Data>(arg.name);
    data->update_input_desc_x(desc);
    return data;
  }
  return nullptr;
}

int16_t float2half(float f) {
  int32_t hs, he, hm, *ptr;
  int16_t rlt;
  ptr = reinterpret_cast<int32_t*>(&f);
  hs = ((*ptr) & 0x80000000) >> 16;
  he = ((*ptr) & 0x7f800000) >> 23;
  he = he - 0x70;
  he = he << 10;
  hm = ((*ptr) & 0x007fffff) >> 13;
  rlt = hs | he | hm;
  return rlt;
}

float half2float(int16_t h) {
  int32_t fs, fe, fm, rlt;
  int16_t* ptr = &h;
  fs = ((*ptr) & 0x8000) << 16;
  fe = ((*ptr) & 0x7c00) >> 10;
  fe = fe + 0x70;
  fe = fe << 23;
  fm = ((*ptr) & 0x03ff) << 13;
  rlt = fs | fe | fm;
  return *(reinterpret_cast<float*>(&rlt));
}

}  // namespace bridge
}  // namespace npu
}  // namespace lite
}  // namespace paddle
