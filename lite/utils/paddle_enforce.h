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

/*
 * This file defines PADDLE_ENFORCE_xx, which helps to adapt the legacy fluid
 * codes.
 */
#pragma once
#include "lite/utils/cp_logging.h"
#include "lite/utils/string.h"

#define PADDLE_ENFORCE(cond, ...) \
  CHECK((cond)) << paddle::lite::string_format("" __VA_ARGS__);
#define PADDLE_ENFORCE_EQ(a, b, ...) \
  CHECK_EQ((a), (b)) << paddle::lite::string_format("" __VA_ARGS__);
#define PADDLE_ENFORCE_LE(a, b, ...) \
  CHECK_LE((a), (b)) << paddle::lite::string_format("" __VA_ARGS__);
#define PADDLE_ENFORCE_LT(a, b, ...) \
  CHECK_LT((a), (b)) << paddle::lite::string_format("" __VA_ARGS__);

#define PADDLE_ENFORCE_GE(a, b, ...) \
  CHECK_GE((a), (b)) << paddle::lite::string_format("" __VA_ARGS__);
#define PADDLE_ENFORCE_GT(a, b, ...) \
  CHECK_GT((a), (b)) << paddle::lite::string_format("" __VA_ARGS__);

#ifndef PADDLE_THROW
#define PADDLE_THROW(...) printf("" __VA_ARGS__);
#endif
