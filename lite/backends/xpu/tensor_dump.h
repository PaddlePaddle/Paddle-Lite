// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once
#include <unistd.h>  // getcwd, access, F_OK
#include <cstring>
#include <memory>
#include <set>
#include <string>
#include <vector>
#include "lite/backends/xpu/npy_dump.h"
#include "lite/backends/xpu/target_wrapper.h"
#include "lite/core/op_lite.h"

namespace paddle {
namespace lite {
namespace xpu {
namespace npy {

const static std::set<std::string> invalid_op_nodes = {  // NOLINT
    "while",
    "conditional_block",
    "conditional_block_infer",
    "subgraph"};                                               // NOLINT
const static std::string target_trans_name = "/target_trans";  // NOLINT

static thread_local int op_count = 0;

bool create_dirs(const std::string& path) {
  uint32_t beginCmpPath = 0;
  uint32_t endCmpPath = 0;
  std::string fullPath = "";

  if ('/' != path[0]) {  // relative path
    // get current path
    fullPath = getcwd(nullptr, 0);
    beginCmpPath = fullPath.size();
    fullPath = fullPath + "/" + path;
  } else {  // absolute path
    fullPath = path;
    beginCmpPath = 1;
  }
  if (fullPath[fullPath.size() - 1] != '/') {
    fullPath += "/";
  }
  endCmpPath = fullPath.size();

  // create dirs
  for (uint32_t i = beginCmpPath; i < endCmpPath; i++) {
    if ('/' == fullPath[i]) {
      std::string curPath = fullPath.substr(0, i);
      if (access(curPath.c_str(), F_OK) != 0) {
        if (mkdir(curPath.c_str(),
                  S_IRUSR | S_IWUSR | S_IXUSR | S_IRWXG | S_IRWXO) == -1) {
          LOG(INFO) << "mkdir: " << curPath << "failed";
          return false;
        }
      }
    }
  }
  return true;
}

template <typename T>
void DumpLog(const T* ptr,
             size_t len,
             const std::string& tensor_name,
             double sum,
             const std::string& dump_log_path = "") {
  LOG(INFO) << "******" << tensor_name << ",len=" << len << ",sum=" << sum
            << ",mean=" << sum / len << "******";

  int print_size = len < 20 ? len : 20;
  if (dump_log_path.empty()) {
    return;
  }

  std::ofstream f;
  f.open(dump_log_path, std::ofstream::out | std::ofstream::app);
  f << "******" << tensor_name << ",len=" << len << ",sum=" << sum
    << ",mean=" << sum / len << "******" << std::endl;

  f << "------";
  for (int i = 0; i < print_size; i++) {
    f << static_cast<float>(*(ptr + i)) << ",";
  }
  f << "------" << std::endl;
}

template <typename T>
void DumpCPUMem(const T* ptr,
                size_t len,
                const std::string& tensor_dump_filename,
                const std::string& tensor_name,
                const std::string& dump_log_path) {
  double sum = 0;
  for (size_t i = 0; i < len; ++i) {
    sum += static_cast<double>(*(ptr + i));
  }

  SaveArrayAsNumpy(tensor_dump_filename, false, len, ptr);
  DumpLog(ptr, len, tensor_name, sum, dump_log_path);
}

template <typename T>
void DumpXPUMem(const T* ptr,
                size_t len,
                const std::string& tensor_dump_filename,
                const std::string& tensor_name,
                const std::string& dump_log_path) {
  std::vector<T> cpu_mem(len, 0);
  TargetWrapperXPU::MemcpySync(
      cpu_mem.data(), ptr, len * sizeof(T), IoDirection::DtoH);

  double sum = 0;
  for (size_t i = 0; i < len; ++i) {
    sum += cpu_mem[i];
  }

  SaveArrayAsNumpy(tensor_dump_filename, false, len, cpu_mem.data());
  DumpLog(cpu_mem.data(), len, tensor_name, sum, dump_log_path);
}

template <typename T>
void DumpTensor(const Tensor* tensor,
                const std::string& tensor_dump_filename,
                const std::string& tensor_name,
                const TargetType& tensor_target,
                const std::string& dump_log_path) {
  if (tensor->template data<T>() == nullptr) {
    LOG(INFO) << "Current tensor:" << tensor_name
              << "is nullptr. so it is not need dump . ";
    return;
  }

  if (tensor->data_size() <= 0) {
    LOG(INFO) << "Current tensor:" << tensor_name
              << " data size <= 0. so it is not need dump . ";
    return;
  }

  try {
    if (tensor_target == TargetType::kXPU) {
      DumpXPUMem<T>(tensor->data<T>(),
                    tensor->data_size(),
                    tensor_dump_filename,
                    tensor_name,
                    dump_log_path);
      return;
    }

    if (tensor_target == TargetType::kX86) {
      DumpCPUMem<T>(tensor->template data<T>(),
                    tensor->data_size(),
                    tensor_dump_filename,
                    tensor_name,
                    dump_log_path);
      return;
    }

    if (tensor_target == TargetType::kHost) {
      DumpCPUMem<T>(tensor->template data<T>(),
                    tensor->data_size(),
                    tensor_dump_filename,
                    tensor_name,
                    dump_log_path);
      return;
    }
  } catch (...) {
    LOG(INFO) << "Current tensor dump error:" << tensor_name;
  }
}

void DumpOpoutTensor(Instruction* inst,
                     std::shared_ptr<OpLite> op,
                     const std::string& dump_tensor_path = "",
                     const std::string& dump_log_path = "") {
  auto op_name = inst->kernel()->op_type();
  LOG(INFO) << "**************"
            << "op_name:" << op_name << ", op_count" << op_count
            << "**************";
  if (invalid_op_nodes.count(op_name)) {
    return;
  }

  std::string dump_path = "";
  if (!dump_tensor_path.empty()) {
    std::stringstream op_count_ss;
    op_count_ss << std::setw(4) << std::setfill('0') << op_count;
    dump_path =
        dump_tensor_path + "/" + op_count_ss.str() + "_" + op_name + "/";
    op_count++;
    bool iscreate = create_dirs(dump_path);
    if (!iscreate) {
      return;
    }
  }

  auto output_names = inst->op()->op_info()->output_names();
  for (auto tensor_name : output_names) {
    std::string tmp;
    CHECK(inst->op()->op_info()->GetOutputArgname(tensor_name, &tmp));
    auto decl_arg_type = inst->kernel()->GetOutputDeclType(tmp);

    // FindTensor API can't use when tensor precision type is kAny.
    if (decl_arg_type->precision() == PrecisionType::kAny) {
      continue;
    }

    auto tensor_target = decl_arg_type->target();

    auto* pout = op->scope()->FindTensor(tensor_name);
    CHECK(pout != nullptr) << "find tensor failed: " << tensor_name;

    std::string name = tensor_name;
    int start_index = name.find(target_trans_name);
    if (start_index >= 0) {
      name.erase(start_index, target_trans_name.size());
    }

    // transform tensor name: A/B  -> A_B,as the file name of the dump tensor.
    std::replace(name.begin(), name.end(), '/', '_');

    std::string tensor_dump_filename = "";
    if (!dump_path.empty()) {
      tensor_dump_filename = dump_path + name + ".npy";
    }

    switch (pout->precision()) {
      case PrecisionType::kFP64: {
        DumpTensor<double>(pout,
                           tensor_dump_filename,
                           tensor_name,
                           tensor_target,
                           dump_log_path);
        break;
      }
      case PrecisionType::kFloat: {
        DumpTensor<float>(pout,
                          tensor_dump_filename,
                          tensor_name,
                          tensor_target,
                          dump_log_path);
        break;
      }
      case PrecisionType::kFP16: {
        DumpTensor<float16>(pout,
                            tensor_dump_filename,
                            tensor_name,
                            tensor_target,
                            dump_log_path);
        break;
      }
      case PrecisionType::kInt8: {
        DumpTensor<int8_t>(pout,
                           tensor_dump_filename,
                           tensor_name,
                           tensor_target,
                           dump_log_path);
        break;
      }

      case PrecisionType::kUInt8: {
        DumpTensor<uint8_t>(pout,
                            tensor_dump_filename,
                            tensor_name,
                            tensor_target,
                            dump_log_path);
        break;
      }

      case PrecisionType::kInt16: {
        DumpTensor<int16_t>(pout,
                            tensor_dump_filename,
                            tensor_name,
                            tensor_target,
                            dump_log_path);
        break;
      }

      case PrecisionType::kInt32: {
        DumpTensor<int32_t>(pout,
                            tensor_dump_filename,
                            tensor_name,
                            tensor_target,
                            dump_log_path);
        break;
      }

      case PrecisionType::kInt64: {
        DumpTensor<int64_t>(pout,
                            tensor_dump_filename,
                            tensor_name,
                            tensor_target,
                            dump_log_path);
        break;
      }

      case PrecisionType::kBool: {
        DumpTensor<int8_t>(pout,
                           tensor_dump_filename,
                           tensor_name,
                           tensor_target,
                           dump_log_path);
        break;
      }

      default: {
        LOG(INFO) << "unsupport Tensor Type ,tensor name:" << tensor_name;
        break;
      }
    }
  }
}

}  // namespace npy
}  // namespace xpu
}  // namespace lite
}  // namespace paddle
