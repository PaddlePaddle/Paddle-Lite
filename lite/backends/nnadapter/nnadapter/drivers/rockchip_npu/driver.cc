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
namespace rockchip_npu {

Model::~Model() {
  if (!execution_) {
    delete execution_;
  }
  if (!graph_) {
    delete graph_;
  }
}

int Model::CreateFromGraph(driver::Graph* graph) {
  graph_ = new rk::nn::Graph();
  if (!graph_) {
    return NNADAPTER_OUT_OF_MEMORY;
  }
  std::vector<Operation*> operations =
      driver::sortOperationsInTopologicalOrder(graph);
  for (auto operation : operations) {
    switch (operation->type) {
      case NNADAPTER_CONV_2D:
        ConvertConv2D(operation);
        break;
      default:
        NNADAPTER_LOG(ERROR) << "Unsupported operation(" << operation->type
                             << ") is found.";
        break;
    }
  }
  std::vector<rk::nn::Tensor *> input_nodes, output_nodes;
  graph_->SetInputsOutputs(input_nodes, output_nodes);
  execution_ = new rk::nn::Exection(graph_);
  execution_->Build();
  return NNADAPTER_NO_ERROR;
}

int Model::CreateFromCache(void* buffer, size_t length) {
  return NNADAPTER_NO_ERROR;
}

int Model::ConvertConv2D(driver::Operation* operation) {
  auto& inputOperands = operation->inputs;
  auto& outputOperands = operation->outputs;
  auto inputCount = inputOperands.size();
  auto outputCount = outputOperands.size();
  NNADAPTER_CHECK_GE(inputCount, 10);
  NNADAPTER_CHECK_EQ(outputCount, 1);

  int oc = 1;
  auto filterOperand = inputOperands[3];
  auto paddingWidthLeft = *reinterpret_cast<int32_t*>(inputOperands[3]->buffer);
  auto paddingWidthRight =
      *reinterpret_cast<int32_t*>(inputOperands[4]->buffer);
  auto paddingHeightTop = *reinterpret_cast<int32_t*>(inputOperands[5]->buffer);
  auto paddingHeightBottom =
      *reinterpret_cast<int32_t*>(inputOperands[6]->buffer);
  auto strideWidth = *reinterpret_cast<int32_t*>(inputOperands[7]->buffer);
  auto strideHeight = *reinterpret_cast<int32_t*>(inputOperands[8]->buffer);
  auto fuseCode = *reinterpret_cast<int32_t*>(inputOperands[9]->buffer);
  int32_t dilationWidth = 1;
  int32_t dilationHeight = 1;
  if (inputCount >= 12) {
    dilationWidth = *reinterpret_cast<int32_t*>(inputOperands[10]->buffer);
    dilationHeight = *reinterpret_cast<int32_t*>(inputOperands[11]->buffer);
  }

  rk::nn::Conv2DAttr attr;
  attr.ksize[0] = filterOperand->type.dimensions[2];
  attr.ksize[1] = filterOperand->type.dimensions[3];
  attr.stride[0] = strideWidth;
  attr.stride[1] = strideHeight;
  attr.pad[0] = paddingWidthLeft;
  attr.pad[1] = paddingWidthRight;
  attr.pad[2] = paddingHeightTop;
  attr.pad[3] = paddingHeightBottom;
  attr.group = 1;
  attr.weights = oc;
  attr.dilation[0] = dilationWidth;
  attr.dilation[1] = dilationHeight;
  attr.pad_type = rk::nn::PadType::AUTO;
  // fuse RELU ?
  if (fuseCode == NNADAPTER_FUSED_NONE) {
    attr.has_relu = false;
  } else if (fuseCode == NNADAPTER_FUSED_RELU) {
    attr.has_relu = true;
  } else {
    NNADAPTER_LOG(ERROR) << "Unsupported fuse_code(" << operation->type
                         << ") is found.";
  }
  // graph->AddOperator(rk::nn::OperatorType::CONV2D, inputs, outputs, &attr);
  return NNADAPTER_NO_ERROR;
}

int createContext(void** context) {
  if (!context) {
    return NNADAPTER_INVALID_PARAMETER;
  }
  auto c = new Context(nullptr);
  if (!c) {
    *context = nullptr;
    NNADAPTER_LOG(ERROR) << "Failed to create context for rockchip_npu.";
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
  NNADAPTER_LOG(INFO) << "Create model from graph for rockchip_npu.";
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
  NNADAPTER_LOG(INFO) << "Create model from cache for rockchip_npu.";
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
    NNADAPTER_LOG(INFO) << "Destroy model for imagination_nna.";
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

}  // namespace rockchip_npu
}  // namespace driver
}  // namespace nnadapter

nnadapter::driver::Driver NNADAPTER_EXPORT
    NNADAPTER_AS_SYM2(NNADAPTER_DRIVER_TARGET) = {
        .name = NNADAPTER_AS_STR2(NNADAPTER_DRIVER_NAME),
        .vendor = "Rockchip",
        .type = NNADAPTER_ACCELERATOR,
        .version = 1,
        .createContext = nnadapter::driver::rockchip_npu::createContext,
        .destroyContext = nnadapter::driver::rockchip_npu::destroyContext,
        .createModelFromGraph =
            nnadapter::driver::rockchip_npu::createModelFromGraph,
        .createModelFromCache =
            nnadapter::driver::rockchip_npu::createModelFromCache,
        .destroyModel = nnadapter::driver::rockchip_npu::destroyModel,
        .runModelSync = nnadapter::driver::rockchip_npu::runModelSync,
        .runModelAsync = nnadapter::driver::rockchip_npu::runModelAsync,
};
