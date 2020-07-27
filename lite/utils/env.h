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

#pragma once
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>

// Specify the path of configuration file for the subgraph segmentation, an
// example is shown as below:
// op_type:in_var_name_0,in_var_name1:out_var_name_0,out_var_name1
// op_type::out_var_name_0
// op_type:in_var_name_0
// op_type
#define SUBGRAPH_CUSTOM_PARTITION_CONFIG_FILE \
  "SUBGRAPH_CUSTOM_PARTITION_CONFIG_FILE"

// The original weight/local/unused variables in the subblock of the subgraph op
// will be saved only if 'SUBGRAPH_ONLINE_MODE' is set to true(default) during
// the analysis phase, it ensure the ops in the subblock can be converted to the
// target device model online during the execution phase.
#define SUBGRAPH_ONLINE_MODE "SUBGRAPH_ONLINE_MODE"

namespace paddle {
namespace lite {

static std::string GetStringFromEnv(const std::string& str,
                                    const std::string& def = "") {
  char* variable = std::getenv(str.c_str());
  if (!variable) {
    return def;
  }
  return std::string(variable);
}

static bool GetBoolFromEnv(const std::string& str, bool def = false) {
  char* variable = std::getenv(str.c_str());
  if (!variable) {
    return def;
  }
  if (strcmp(variable, "false") == 0 || strcmp(variable, "0") == 0) {
    return false;
  } else {
    return true;
  }
}

static int GetIntFromEnv(const std::string& str, int def = 0) {
  char* variable = std::getenv(str.c_str());
  if (!variable) {
    return def;
  }
  return atoi(variable);
}

static double GetDoubleFromEnv(const std::string& str, double def = 0.0) {
  char* variable = std::getenv(str.c_str());
  if (!variable) {
    return def;
  }
  return atof(variable);
}

static uint64_t GetUInt64FromEnv(const std::string& str, uint64_t def = 0ul) {
  char* variable = std::getenv(str.c_str());
  if (!variable) {
    return def;
  }
  return static_cast<uint64_t>(atol(variable));
}

}  // namespace lite
}  // namespace paddle
