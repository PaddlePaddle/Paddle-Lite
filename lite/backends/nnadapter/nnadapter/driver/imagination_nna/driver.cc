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

#include <map>
#include <vector>
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/micros.h"
#include "utility/modeling.h"

namespace nnadapter {
namespace imagination_nna {

class Context {
 public:
  Context() {}
  ~Context() {}

 private:
  void* context_{nullptr};
};

class Program {
 public:
  explicit Program(Context* context) : context_(context) {}
  ~Program() {}

  int Build(hal::Model* model, hal::Cache* cache) {
    std::vector<hal::Operation*> operations =
        SortOperationsInTopologicalOrder(model);
    for (auto operation : operations) {
      switch (operation->type) {
        case NNADAPTER_CONV_2D:
        default:
          NNADAPTER_LOG(FATAL) << "Unsupported operation("
                               << OperationTypeToString(operation->type)
                               << ") is found.";
          break;
      }
    }
    return NNADAPTER_NO_ERROR;
  }

 private:
  Context* context_{nullptr};
  std::map<hal::Operand*, int> nodes_;
};

int CreateContext(void** context) {
  if (!context) {
    return NNADAPTER_INVALID_PARAMETER;
  }
  auto c = new Context();
  if (!c) {
    *context = nullptr;
    NNADAPTER_LOG(FATAL) << "Failed to create context for imagination_nna.";
    return NNADAPTER_OUT_OF_MEMORY;
  }
  *context = reinterpret_cast<void*>(c);
  return NNADAPTER_NO_ERROR;
}

void DestroyContext(void* context) {
  if (!context) {
    auto c = reinterpret_cast<Context*>(context);
    delete c;
  }
}

int CreateProgram(void* context,
                  hal::Model* model,
                  hal::Cache* cache,
                  void** program) {
  NNADAPTER_LOG(INFO) << "Create program for imagination_nna.";
  if (!context || !(model && cache) || !program) {
    return NNADAPTER_INVALID_PARAMETER;
  }
  *program = nullptr;
  auto c = reinterpret_cast<Context*>(context);
  auto p = new Program(c);
  if (!p) {
    return NNADAPTER_OUT_OF_MEMORY;
  }
  int result = p->Build(model, cache);
  if (result == NNADAPTER_NO_ERROR) {
    *program = reinterpret_cast<void*>(p);
  }
  return result;
}

void DestroyProgram(void* program) {
  if (program) {
    NNADAPTER_LOG(INFO) << "Destroy program for imagination_nna.";
    auto p = reinterpret_cast<Program*>(program);
    delete p;
  }
}

int ExecuteProgram(void* program,
                   uint32_t input_count,
                   hal::Argument* input_arguments,
                   uint32_t output_count,
                   hal::Argument* output_arguments) {
  if (!program || !output_arguments || !output_count) {
    return NNADAPTER_INVALID_PARAMETER;
  }
  auto p = reinterpret_cast<Program*>(program);
  return NNADAPTER_NO_ERROR;
}

}  // namespace imagination_nna
}  // namespace nnadapter

NNADAPTER_EXPORT nnadapter::hal::Device NNADAPTER_AS_SYM2(
    NNADAPTER_DRIVER_NAME) = {
    .name = NNADAPTER_AS_STR2(NNADAPTER_DEVICE_NAME),
    .vendor = "Imagination",
    .type = NNADAPTER_ACCELERATOR,
    .version = 1,
    .create_context = nnadapter::imagination_nna::CreateContext,
    .destroy_context = nnadapter::imagination_nna::DestroyContext,
    .create_program = nnadapter::imagination_nna::CreateProgram,
    .destroy_program = nnadapter::imagination_nna::DestroyProgram,
    .execute_program = nnadapter::imagination_nna::ExecuteProgram,
};
