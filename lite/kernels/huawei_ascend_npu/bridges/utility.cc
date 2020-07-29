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

#include "lite/kernels/huawei_ascend_npu/bridges/utility.h"
#include <utility>

namespace paddle {
namespace lite {
namespace subgraph {
namespace huawei_ascend_npu {

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

ge::DataType CvtPrecisionType(PrecisionType itype) {
  ge::DataType otype = ge::DT_FLOAT;
  switch (itype) {
    case PRECISION(kFloat):
      otype = ge::DT_FLOAT;
      break;
    case PRECISION(kFP16):
      otype = ge::DT_FLOAT16;
      break;
    case PRECISION(kInt8):
      otype = ge::DT_INT8;
      break;
    case PRECISION(kInt16):
      otype = ge::DT_INT16;
      break;
    case PRECISION(kInt32):
      otype = ge::DT_INT32;
      break;
    case PRECISION(kInt64):
      otype = ge::DT_INT64;
      break;
    // TODO(liq27) support more precision type
    default:
      LOG(FATAL) << "[HUAWEI_ASCEND_NPU] Can not convert precision type("
                 << PrecisionToStr(itype) << ") from Lite to NPU";
      break;
  }
  return otype;
}

ge::Format CvtDataLayoutType(DataLayoutType itype) {
  ge::Format otype = ge::FORMAT_NCHW;
  switch (itype) {
    case DATALAYOUT(kNCHW):
      otype = ge::FORMAT_NCHW;
      break;
    case DATALAYOUT(kNHWC):
      otype = ge::FORMAT_NHWC;
      break;
    // TODO(liq27) support more data layout type
    default:
      LOG(FATAL) << "[HUAWEI_ASCEND_NPU] Can not convert data layout type("
                 << DataLayoutToStr(itype)
                 << ") from Lite to HUAWEI_ASCEND_NPU";
      break;
  }
  return otype;
}

std::vector<int64_t> CvtShape(const std::vector<int64_t>& in_shape) {
  std::vector<int64_t> out_shape;
  // Padding the shape to 4-dimensions(NCHW)
  for (size_t i = 0; i < 4 - in_shape.size(); i++) {
    out_shape.push_back(1);
  }
  for (size_t i = 0; i < in_shape.size(); i++) {
    out_shape.push_back(in_shape[i]);
  }
  return out_shape;
}

std::vector<int64_t> CvtShape(const DDim& in_dims) {
  return CvtShape(in_dims.Vectorize());
}

ge::Tensor CvtTensor(const Tensor& in_tensor,
                     std::vector<int64_t> out_shape,
                     DataLayoutType in_layout) {
  PrecisionType in_precision = in_tensor.precision();
  auto in_size = in_tensor.dims().production();
  auto in_shape = in_tensor.dims().Vectorize();
  if (out_shape.empty()) {
    out_shape = in_shape;
  }
  ge::TensorDesc out_desc(ge::Shape(out_shape),
                          CvtDataLayoutType(in_layout),
                          CvtPrecisionType(in_precision));
  auto out_size = out_desc.GetShape().GetShapeSize();
  CHECK_EQ(out_size, in_size);
  ge::Tensor out_tensor;
  out_tensor.SetTensorDesc(out_desc);
  out_tensor.SetData(reinterpret_cast<const uint8_t*>(in_tensor.raw_data()),
                     in_tensor.memory_size());
  return out_tensor;
}

int CvtActMode(std::string act_type) {
  int act_mode = 1;
  if (act_type == "sigmoid") {
    act_mode = 0;
  } else if (act_type == "relu") {
    act_mode = 1;
  } else if (act_type == "tanh") {
    act_mode = 2;
  } else if (act_type == "relu_clipped" || act_type == "relu6") {
    act_mode = 3;
  } else if (act_type == "elu") {
    act_mode = 4;
  } else if (act_type == "leaky_relu") {
    act_mode = 5;
  } else if (act_type == "abs") {
    act_mode = 6;
  } else if (act_type == "softsign") {
    act_mode = 8;
  } else if (act_type == "softplus") {
    act_mode = 9;
  } else if (act_type == "hard_sigmoid") {
    act_mode = 10;
  } else if (act_type == "thresholded_relu") {
    act_mode = 11;
  } else {
    // TODO(liqi27) support more activation mode
    LOG(FATAL) << "[HUAWEI_ASCEND_NPU] Unsupported activation type "
               << act_type;
  }
  return act_mode;
}

const std::string& CvtFormat(ge::Format format) {
  static const int MAX_FORMAT_LENGTH = 25;
  static const std::string format2string[] = {
      "FORMAT_NCHW = 0",
      "FORMAT_NHWC = 1",
      "FORMAT_ND = 2",
      "FORMAT_NC1HWC0 = 3",
      "FORMAT_FRACTAL_Z = 4",
      "FORMAT_NC1C0HWPAD = 5",
      "FORMAT_NHWC1C0 = 6",
      "FORMAT_FSR_NCHW = 7",
      "FORMAT_FRACTAL_DECONV = 8",
      "FORMAT_C1HWNC0 = 9",
      "FORMAT_FRACTAL_DECONV_TRANSPOSE = 10",
      "FORMAT_FRACTAL_DECONV_SP_STRIDE_TRANS = 11",
      "FORMAT_NC1HWC0_C04 = 12",
      "FORMAT_FRACTAL_Z_C04 = 13",
      "FORMAT_CHWN = 14",
      "FORMAT_FRACTAL_DECONV_SP_STRIDE8_TRANS = 15",
      "FORMAT_HWCN = 16",
      "FORMAT_NC1KHKWHWC0 = 17",
      "FORMAT_BN_WEIGHT = 18",
      "FORMAT_FILTER_HWCK = 19",
      "FORMAT_HASHTABLE_LOOKUP_LOOKUPS = 20",
      "FORMAT_HASHTABLE_LOOKUP_KEYS = 21",
      "FORMAT_HASHTABLE_LOOKUP_VALUE = 22",
      "FORMAT_HASHTABLE_LOOKUP_OUTPUT = 23",
      "FORMAT_HASHTABLE_LOOKUP_HITS = 24"};
  auto x = static_cast<int>(format);
  CHECK_LT(x, MAX_FORMAT_LENGTH);
  return format2string[x];
}

const std::string& CvtDataType(ge::DataType data_type) {
  static const int MAX_DATATYPE_LENGTH = 14;
  static const std::string datatype2string[] = {"DT_FLOAT=0",
                                                "DT_FLOAT16=1",
                                                "DT_INT8=2",
                                                "DT_INT32=3",
                                                "DT_UINT8=4",
                                                "Unknown=5",
                                                "DT_INT16=6",
                                                "DT_UINT16=7",
                                                "DT_UINT32=8",
                                                "DT_INT64=9",
                                                "DT_UINT64=10",
                                                "DT_DOUBLE=11",
                                                "DT_BOOL=12",
                                                "DT_STRING=13"};

  auto x = static_cast<int>(data_type);
  CHECK_LT(x, MAX_DATATYPE_LENGTH);
  return datatype2string[x];
}

}  // namespace huawei_ascend_npu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle
