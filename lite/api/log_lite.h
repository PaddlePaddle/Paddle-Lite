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

#pragma once

#define LOGI(fmt, ...) printf(fmt, ##__VA_ARGS__)
#define LOGE(fmt, ...) printf(fmt, ##__VA_ARGS__)
#define LOGF(fmt, ...)        \
  printf(fmt, ##__VA_ARGS__); \
  exit(1)

#define LCHECK(a, fmt, ...)     \
  do {                          \
    if (!a) {                   \
      LOGF(fmt, ##__VA_ARGS__); \
    }                           \
  } while (0)

#define LCHECK_EQ(a, b, fmt, ...) \
  do {                            \
    if (a != b) {                 \
      LOGF(fmt, ##__VA_ARGS__);   \
    }                             \
  } while (0)

#define LCHECK_NE(a, b, fmt, ...) \
  do {                            \
    if (a == b) {                 \
      LOGF(fmt, ##__VA_ARGS__);   \
    }                             \
  } while (0)

#define LCHECK_GE(a, b, fmt, ...) \
  do {                            \
    if (a < b) {                  \
      LOGF(fmt, ##__VA_ARGS__);   \
    }                             \
  } while (0)

#define LCHECK_GT(a, b, fmt, ...) \
  do {                            \
    if (a <= b) {                 \
      LOGF(fmt, ##__VA_ARGS__);   \
    }                             \
  } while (0)

#define LCHECK_LE(a, b, fmt, ...) \
  do {                            \
    if (a > b) {                  \
      LOGF(fmt, ##__VA_ARGS__);   \
    }                             \
  } while (0)

#define LCHECK_LT(a, b, fmt, ...) \
  do {                            \
    if (a >= b) {                 \
      LOGF(fmt, ##__VA_ARGS__);   \
    }                             \
  } while (0)
