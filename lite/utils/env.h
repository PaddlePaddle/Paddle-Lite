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
#include <vector>
#include "lite/utils/io.h"
#include "lite/utils/log/cp_logging.h"

// The environment variables for the subgraph settings, use "SUBGRAPH_" as
// prefix.
// Specify the configuration file path or buffer for the subgraph segmentation,
// it lists the operators that are forced to use the CPU, an example is shown as
// below:
// op_type:in_var_name_0,in_var_name1:out_var_name_0,out_var_name1
// op_type::out_var_name_0
// op_type:in_var_name_0
// op_type
#define SUBGRAPH_PARTITION_CONFIG_FILE "SUBGRAPH_PARTITION_CONFIG_FILE"
#define SUBGRAPH_PARTITION_CONFIG_BUFFER "SUBGRAPH_PARTITION_CONFIG_BUFFER"

// The original weight/local/unused variables in the subblock of the subgraph op
// will be saved only if 'SUBGRAPH_ONLINE_MODE' is set to true(default) during
// the analysis phase, it ensure the ops in the subblock can be converted to the
// target device model online during the execution phase.
#define SUBGRAPH_ONLINE_MODE "SUBGRAPH_ONLINE_MODE"

// The environment variables for the opencl memory config settings
// Specify the path of configuration file for the opencl buffer memory config,
// an
// example is shown as below:
// op_type:in_var_name_0,in_var_name1:out_var_name_0,out_var_name1
// op_type::out_var_name_0
// op_type:in_var_name_0
// op_type
#define OPENCL_MEMORY_CONFIG_FILE "OPENCL_MEMORY_CONFIG_FILE"

// Due to various reasons (such as bugs from PaddleSlim), some ops in the model
// lack quantization parameters. Optionally, the missing quantization parameters
// can be completed by the following rules.
// (a) Complete the output scale from the input scale of its consumer ops.
// (b) Complete the output scale from the user-defined configurations.
// (c) Complete the output scale from its out_threshold attribute.
// (d) Complete the input scale from the output scale of its producer op.
// (e) Complete the output scale according to the input scale, or complete the
// input scale according to the output scale, because the input scale and output
// scale of some ops should be the same.
// (f) Complete the output scale according to the formula of some special ops
// themselves.
// QUANT_AUTO_COMPLETE_SCALE_LEVEL support the following level:
// "0", default to apply the rule (a)(c)(d);
// "1", apply the rule (a)(c)(d) and set the output scale even if the op has no
// out_thresold attribute;
// "2", apply the rule (a)(c)(d)(e) and set the output scale even if the op has
// no out_thresold attribute;
// "3", apply the rule (a)(c)(d)(e)(f) and set the output scale even if the op
// has no out_thresold attribute;
#define QUANT_AUTO_COMPLETE_SCALE_LEVEL "QUANT_AUTO_COMPLETE_SCALE_LEVEL"
// Specify the configuration file path or data to apply the rule (b).
#define QUANT_AUTO_COMPLETE_SCALE_CONFIG_FILE \
  "QUANT_AUTO_COMPLETE_SCALE_CONFIG_FILE"
#define QUANT_AUTO_COMPLETE_SCALE_CONFIG_BUFFER \
  "QUANT_AUTO_COMPLETE_SCALE_CONFIG_BUFFER"

// Specify the configuration file path or buffer for the mixed precision
// quantization, it lists the operators that enforce fp32 precision, an example
// is shown as below:
// op_type:in_var_name_0,in_var_name1:out_var_name_0,out_var_name1
// op_type::out_var_name_0
// op_type:in_var_name_0
// op_type
#define MIXED_PRECISION_QUANTIZATION_CONFIG_FILE \
  "MIXED_PRECISION_QUANTIZATION_CONFIG_FILE"
#define MIXED_PRECISION_QUANTIZATION_CONFIG_BUFFER \
  "MIXED_PRECISION_QUANTIZATION_CONFIG_BUFFER"

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

static std::string GetConfigsFromEnv(const std::string& str1,
                                     const std::string& str2,
                                     const std::string& def = "") {
  std::string configs;
  auto path = GetStringFromEnv(str1);
  if (!path.empty()) {
    std::vector<char> buffer;
    if (ReadFile(path, &buffer, false)) {
      if (!buffer.empty()) {
        configs.insert(configs.begin(), buffer.begin(), buffer.end());
      }
    } else {
      LOG(WARNING) << "Missing the config file " << path;
    }
  }
  if (configs.empty()) {
    configs = GetStringFromEnv(str2);
  }
  return configs;
}

}  // namespace lite
}  // namespace paddle
