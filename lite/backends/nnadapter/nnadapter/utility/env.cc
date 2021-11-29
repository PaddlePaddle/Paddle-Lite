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

#include "utility/env.h"
#include "utility/micros.h"

namespace nnadapter {

NNADAPTER_EXPORT std::string GetStringFromEnv(const std::string& str,
                                              const std::string& def) {
  char* variable = std::getenv(str.c_str());
  if (!variable) {
    return def;
  }
  return std::string(variable);
}

NNADAPTER_EXPORT bool GetBoolFromEnv(const std::string& str, bool def) {
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

NNADAPTER_EXPORT int GetIntFromEnv(const std::string& str, int def) {
  char* variable = std::getenv(str.c_str());
  if (!variable) {
    return def;
  }
  return atoi(variable);
}

NNADAPTER_EXPORT double GetDoubleFromEnv(const std::string& str, double def) {
  char* variable = std::getenv(str.c_str());
  if (!variable) {
    return def;
  }
  return atof(variable);
}

NNADAPTER_EXPORT uint64_t GetUInt64FromEnv(const std::string& str,
                                           uint64_t def) {
  char* variable = std::getenv(str.c_str());
  if (!variable) {
    return def;
  }
  return static_cast<uint64_t>(atol(variable));
}

}  // namespace nnadapter
