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

#include "driver.h"  // NOLINT
#include <vector>
#include "../../nnadapter_logging.h"  // NOLINT

namespace nnadapter {
namespace driver {
namespace mediatek_apu {

Model::~Model() {}

int Model::CreateFromGraph(driver::Graph* graph) {
  std::vector<Operation*> operations =
      driver::sortOperationsInTopologicalOrder(graph);
  for (auto operation : operations) {
    switch (operation->type) {
      case NNADAPTER_CONV_2D:
      default:
        NNADAPTER_LOG(ERROR) << "Unsupported operation(" << operation->type
                             << ") is found.";
        break;
    }
  }
  return NNADAPTER_NO_ERROR;
}

int Model::CreateFromCache(void* buffer, size_t length) {
  return NNADAPTER_NO_ERROR;
}

int createContext(void** context) {
  if (!context) {
    return NNADAPTER_INVALID_PARAMETER;
  }
  auto c = new Context(nullptr);
  if (!c) {
    *context = nullptr;
    NNADAPTER_LOG(ERROR) << "Failed to create context for mediatek_apu.";
    return NNADAPTER_OUT_OF_MEMORY;
  }
  *context = reinterpret_cast<void*>(c);
  return NNADAPTER_NO_ERROR;
}

void destroyContext(void* context) {
  if (!context) {
    auto c = reinterpret_cast<Context*>(context);
    delete c;
  }
}

int createModelFromGraph(void* context, driver::Graph* graph, void** model) {
  NNADAPTER_LOG(INFO) << "Create model from graph for mediatek_apu.";
  if (!context || !graph || !model) {
    return NNADAPTER_INVALID_PARAMETER;
  }
  *model = nullptr;
  auto m = new Model();
  if (!m) {
    return NNADAPTER_OUT_OF_MEMORY;
  }
  int result = m->CreateFromGraph(graph);
  if (result == NNADAPTER_NO_ERROR) {
    *model = reinterpret_cast<void*>(m);
  }
  return result;
}

int createModelFromCache(void* context,
                         void* buffer,
                         size_t length,
                         void** model) {
  if (!context || !buffer || !length || !model) {
    return NNADAPTER_INVALID_PARAMETER;
  }
  NNADAPTER_LOG(INFO) << "Create model from cache for mediatek_apu.";
  *model = nullptr;
  auto m = new Model();
  if (!m) {
    return NNADAPTER_OUT_OF_MEMORY;
  }
  int result = m->CreateFromCache(buffer, length);
  if (result == NNADAPTER_NO_ERROR) {
    *model = reinterpret_cast<void*>(m);
  }
  return NNADAPTER_NO_ERROR;
}

void destroyModel(void* context, void* model) {
  if (context && model) {
    NNADAPTER_LOG(INFO) << "Destroy model for mediatek_apu.";
    auto m = reinterpret_cast<Model*>(model);
    delete m;
  }
}

int runModelSync(void* context,
                 void* model,
                 uint32_t inputCount,
                 Operand** inputs,
                 uint32_t outputCount,
                 Operand** outputs) {
  if (!context || !model || !outputs || !inputCount) {
    return NNADAPTER_INVALID_PARAMETER;
  }
  auto m = reinterpret_cast<Model*>(model);
  return NNADAPTER_NO_ERROR;
}

int runModelAsync(void* context,
                  void* model,
                  uint32_t inputCount,
                  Operand** inputs,
                  uint32_t outputCount,
                  Operand** outputs) {
  return NNADAPTER_NO_ERROR;
}

}  // namespace mediatek_apu
}  // namespace driver
}  // namespace nnadapter

nnadapter::driver::Driver NNADAPTER_EXPORT
    NNADAPTER_AS_SYM2(NNADAPTER_DRIVER_TARGET) = {
        .name = NNADAPTER_AS_STR2(NNADAPTER_DRIVER_NAME),
        .vendor = "MediaTek",
        .type = NNADAPTER_ACCELERATOR,
        .version = 1,
        .createContext = nnadapter::driver::mediatek_apu::createContext,
        .destroyContext = nnadapter::driver::mediatek_apu::destroyContext,
        .createModelFromGraph =
            nnadapter::driver::mediatek_apu::createModelFromGraph,
        .createModelFromCache =
            nnadapter::driver::mediatek_apu::createModelFromCache,
        .destroyModel = nnadapter::driver::mediatek_apu::destroyModel,
        .runModelSync = nnadapter::driver::mediatek_apu::runModelSync,
        .runModelAsync = nnadapter::driver::mediatek_apu::runModelAsync,
};
