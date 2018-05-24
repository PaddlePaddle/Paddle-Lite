/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#ifdef PADDLE_MOBILE_DEBUG
#include <stdio.h>
#include <exception>
#include <sstream>
#include <stdexcept>
#include <string>

#endif

namespace paddle_mobile {

#ifdef PADDLE_MOBILE_DEBUG
struct PaddleMobileException : public std::exception {
  const std::string exception_prefix = "paddle mobile C++ Exception: \n";
  std::string message;

  PaddleMobileException(const char *header, const char *detail,
                        const char *file, const int line) {
    std::stringstream ss;
    ss << exception_prefix << "| " << header << "\n";
    ss << "| [in file] : " << file << " \n";
    ss << "| [on line] : " << line << " \n";
    ss << "| [detail]  : " << detail;
    message = ss.str();
  }
  const char *what() const noexcept { return message.c_str(); }
};

#define PADDLE_MOBILE_THROW_EXCEPTION(...)                                 \
  {                                                                        \
    char buffer[1000];                                                     \
    snprintf(buffer, sizeof(buffer), __VA_ARGS__);                         \
    std::string detail(buffer);                                            \
    throw paddle_mobile::PaddleMobileException("Custom Exception", buffer, \
                                               __FILE__, __LINE__);        \
  }

#define PADDLE_MOBILE_ASSERT(stat, ...)                                       \
  {                                                                           \
    if (stat) {                                                               \
    } else {                                                                  \
      char buffer[1000];                                                      \
      snprintf(buffer, sizeof(buffer), __VA_ARGS__);                          \
      std::string detail(buffer);                                             \
      throw paddle_mobile::PaddleMobileException("paddle-mobile assert",      \
                                                 buffer, __FILE__, __LINE__); \
    }                                                                         \
  }
#else
#define PADDLE_MOBILE_THROW_EXCEPTION(...)
#define PADDLE_MOBILE_ASSERT(stat, ...)
#endif

}  // namespace paddle_mobile