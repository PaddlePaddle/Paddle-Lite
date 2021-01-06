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

/////////////////////////////////////////////////////////////////////////////////
// Auther: DannyIsFunny (github)
// Date:   20201124
// Description: Version defination for Op and corresponding kernels.
/////////////////////////////////////////////////////////////////////////////////

#pragma once
#include <map>
#include <string>
#include "lite/utils/cp_logging.h"
namespace paddle {
namespace lite {

///////////////////////////////////////////////////////////
// Name: KernelVersion
// Description: Version of Paddle-Lite kernel
//              (a list of OpVersion)
///////////////////////////////////////////////////////////

class KernelVersion {
 public:
  // Fill op_versions into kernel_version
  void AddOpVersion(const std::string& name, int32_t op_version) {
    if (!op_versions_.count(name)) {
      op_versions_[name] = op_version;
    } else {
      LOG(FATAL) << "Error: binding kernel to the version of op(" << name
                 << ") more than once is not allowed.";
    }
  }
  // Return the content of kernel_version: list(op_version)
  const std::map<std::string, int32_t>& OpVersions() const {
    return op_versions_;
  }
  // Judge if an op_version has been bound to this kernel.
  bool HasOpVersion(const std::string& op_name) {
    return op_versions_.count(op_name);
  }

  // Get a inner op_version according to op_name.
  int32_t GetOpVersion(const std::string& op_name) {
    if (HasOpVersion(op_name)) {
      return op_versions_[op_name];
    } else {
      LOG(FATAL) << "Error: This kernel has not been bound to Paddle op ("
                 << op_name << ") 's version.";
      return -1;
    }
  }

 private:
  // Paddle OpVersion: Version of Paddle operator
  //              (op_name + version_id)
  std::map<std::string, int32_t> op_versions_;
};

}  // namespace lite
}  // namespace paddle
