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

#include "lite/backends/xpu/builder.h"
#include <mutex>  // NOLINT
#include <utility>
#include "lite/backends/xpu/runtime.h"

namespace paddle {
namespace lite {
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
      LOG(FATAL) << "Can not convert precision type(" << PrecisionToStr(in_type)
                 << ") from Lite to XPU";
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
      LOG(FATAL) << "Can not convert data type(" << PrecisionToStr(in_type)
                 << ") from Lite to XPU";
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

std::shared_ptr<xtcl::xNDArray> CvtTensor(lite::Tensor* in_tensor,
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
  auto out_tensor = std::make_shared<xtcl::xNDArray>(
      xtcl::xNDArray::Empty(out_shape, CvtDataType(in_ptype), {kDLCPU, 0}));
  auto out_data =
      reinterpret_cast<uint8_t*>(out_tensor->ToDLPack()->dl_tensor.data);
  std::memcpy(out_data, in_data, in_bytes);
  return out_tensor;
}

// Build IR graph to model, and store model data into lite tensor
bool BuildModel(std::shared_ptr<xtcl::network::xNetworkBuilder> network_builder,
                std::shared_ptr<xtcl::network::xTensorCompiler::ParamNDArrayMap>
                    const_tensors,
                std::vector<std::shared_ptr<xtcl::xExpr>>* output_layers,
                lite::Tensor* model_data) {
  LOG(INFO) << "[XPU] Build Model.";
  CHECK(network_builder != nullptr);
  CHECK_GT(output_layers->size(), 0);
  CHECK(model_data != nullptr);
  // build network and fill all of constant params
  xtcl::xNetwork network_data =
      network_builder->FinalizeNetwork(*((*output_layers)[0]));
  auto device_target = xtcl::Target::Create("llvm");
  auto model_compiler =
      xtcl::network::xTensorCompiler(network_data, device_target);
  model_compiler.SetParams(*const_tensors);
  model_compiler.Build();
  // register model runtime
  auto model_runtime = std::make_shared<xtcl::network::xRuntimeInstance>(
      model_compiler.CreateRuntimeInstance());
  if (model_runtime == nullptr) {
    LOG(WARNING) << "[XPU] Build Model failed!";
    return false;
  }
  std::string model_name = UniqueName("xpu_model_name");
  DeviceInfo::Global().Insert(model_name, model_runtime);
  model_data->Resize({static_cast<int64_t>(model_name.length() + 1)});
  memcpy(model_data->mutable_data<int8_t>(),
         reinterpret_cast<const int8_t*>(model_name.c_str()),
         model_name.length() + 1);
  return true;
}

}  // namespace xpu
}  // namespace lite
}  // namespace paddle
