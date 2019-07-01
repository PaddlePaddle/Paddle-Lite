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

#include <map>
#include <string>
#include <vector>

#include "./test_helper.h"
#include "common/enforce.h"
#include "common/util.h"
#include "common/log.h"
#include "executor_for_test.h"
#include "framework/ddim.h"
#include "framework/lod_tensor.h"
#include "framework/operator.h"
#include "framework/program/block_desc.h"
#include "framework/program/program.h"
#include "framework/program/program_desc.h"
#include "framework/scope.h"
#include "framework/tensor.h"
#include "framework/variable.h"
#include "io/paddle_mobile.h"
