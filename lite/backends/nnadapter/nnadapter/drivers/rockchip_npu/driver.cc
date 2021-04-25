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
#include <memory>
#include <vector>
#include "../../nnadapter_logging.h"  // NOLINT

namespace nnadapter {
namespace driver {
namespace rockchip_npu {

Program::~Program() {
  if (!execution_) {
    delete execution_;
  }
  if (!graph_) {
    delete graph_;
  }
}

int Program::Build(driver::Model* model, driver::Cache* cache) {
  graph_ = new rk::nn::Graph();
  if (!graph_) {
    return NNADAPTER_OUT_OF_MEMORY;
  }
  std::vector<Operation*> operations =
      driver::SortOperationsInTopologicalOrder(model);
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
  std::vector<std::shared_ptr<rk::nn::Tensor>> input_nodes, output_nodes;
  graph_->SetInputsOutputs(input_nodes, output_nodes);
  execution_ = new rk::nn::Exection(graph_);
  execution_->Build();
  return NNADAPTER_NO_ERROR;
}

int Program::ConvertConv2D(driver::Operation* operation) {
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

int CreateContext(void** context) {
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

void DestroyContext(void* context) {
  if (!context) {
    auto c = reinterpret_cast<Context*>(context);
    delete c;
  }
}

int CreateProgram(void* context,
                  driver::Model* model,
                  driver::Cache* cache,
                  void** program) {
  NNADAPTER_LOG(INFO) << "Create program for rockchip_npu.";
  if (!context || !(model && cache) || !program) {
    return NNADAPTER_INVALID_PARAMETER;
  }
  *program = nullptr;
  auto p = new Program();
  if (!p) {
    return NNADAPTER_OUT_OF_MEMORY;
  }
  int result = p->Build(model, cache);
  if (result == NNADAPTER_NO_ERROR) {
    *program = reinterpret_cast<void*>(p);
  }
  return result;
}

void DestroyProgram(void* context, void* program) {
  if (context && program) {
    NNADAPTER_LOG(INFO) << "Destroy program for rockchip_npu.";
    auto p = reinterpret_cast<Program*>(program);
    delete p;
  }
}

int ExecuteProgram(void* context,
                   void* program,
                   uint32_t input_count,
                   driver::Argument* inputs,
                   uint32_t output_count,
                   driver::Argument* outputs) {
  if (!context || !program || !outputs || !output_count) {
    return NNADAPTER_INVALID_PARAMETER;
  }
  auto p = reinterpret_cast<Program*>(program);
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
        .create_context = nnadapter::driver::rockchip_npu::CreateContext,
        .destroy_context = nnadapter::driver::rockchip_npu::DestroyContext,
        .create_program = nnadapter::driver::rockchip_npu::CreateProgram,
        .destroy_program = nnadapter::driver::rockchip_npu::DestroyProgram,
        .execute_program = nnadapter::driver::rockchip_npu::ExecuteProgram,
};
