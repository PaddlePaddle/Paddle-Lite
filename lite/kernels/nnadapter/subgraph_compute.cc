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

#include "lite/kernels/nnadapter/subgraph_compute.h"
#include <sys/time.h>
#include <time.h>
#include <utility>
#include "lite/core/op_registry.h"
#include "lite/kernels/nnadapter/bridges/converter.h"
#include "lite/kernels/nnadapter/bridges/paddle_use_bridges.h"
#include "lite/kernels/nnadapter/bridges/utility.h"
#include "lite/utils/io.h"
#include "lite/utils/md5.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace nnadapter {

// A simple token for identifying the model cache is generated based on the MD5
// algorithm and the following information: 1) The valid device names 2) The
// input variable names 3) The input variable shapes
std::string GenerateModelCacheToken(
    const std::vector<std::string>& device_names,
    const std::vector<std::string>& input_names,
    const std::vector<std::vector<int64_t>>& input_shapes) {
  std::ostringstream os;
  for (auto device_name : device_names) {
    os << device_name;
  }
  CHECK_EQ(input_names.size(), input_shapes.size());
  for (size_t i = 0; i < input_names.size(); i++) {
    os << input_names[i];
    for (auto input_shape : input_shapes[i]) {
      os << input_shape;
    }
  }
  return MD5(os.str());
}

void* AccessModelInput(void* memory, NNAdapterOperandType* type) {
  CHECK(memory);
  CHECK(type);
  auto tensor = static_cast<Tensor*>(memory);
  // Fill the dimensions and get the host buffer address of model inputs
  subgraph::nnadapter::ConvertDimensions(
      tensor->dims(), type->dimensions, &type->dimension_count);
  return tensor->raw_data();
}

void* AccessModelOutput(void* memory, NNAdapterOperandType* type) {
  CHECK(memory);
  CHECK(type);
  auto tensor = static_cast<Tensor*>(memory);
  auto precision = subgraph::nnadapter::ConvertPrecision(type->precision);
  auto dimensions = subgraph::nnadapter::ConvertDimensions(
      type->dimensions, type->dimension_count);
  tensor->Resize(dimensions);
#define TENSOR_MUTABLE_DATA(ptype, dtype) \
  case PRECISION(ptype):                  \
    tensor->mutable_data<dtype>();        \
    break;
  switch (precision) {
    TENSOR_MUTABLE_DATA(kInt8, int8_t)
    TENSOR_MUTABLE_DATA(kInt32, int32_t)
    TENSOR_MUTABLE_DATA(kInt64, int64_t)
    TENSOR_MUTABLE_DATA(kFloat, float)
    default:
      LOG(ERROR) << "Failed to mutable data for the precsion type("
                 << PrecisionToStr(precision) << ") at output@0x"
                 << string_format("%x", memory) << "!";
      break;
  }
#undef TENSOR_MUTABLE_DATA
  return tensor->raw_data();
}

Program::~Program() {
  if (execution_) {
    NNAdapterExecution_destroy_invoke(execution_);
  }
  if (compilation_) {
    NNAdapterCompilation_destroy_invoke(compilation_);
  }
  if (model_) {
    NNAdapterModel_destroy_invoke(model_);
  }
}

bool Program::LoadFromCache(const std::string& model_cache_token,
                            std::vector<char>* model_cache_buffer,
                            const std::string& model_cache_dir) {
  CHECK(!model_cache_token.empty());
  // Compiling the model to the device-specific programs from the model cache
  // buffer
  int result = NNAdapterCompilation_create_invoke(nullptr,
                                                  model_cache_token.c_str(),
                                                  model_cache_buffer->data(),
                                                  model_cache_buffer->size(),
                                                  model_cache_dir.c_str(),
                                                  context_,
                                                  &compilation_);
  if (result != NNADAPTER_NO_ERROR) {
    LOG(WARNING)
        << "Failed to create a compilation from the model cache buffer ("
        << result << ") !";
    return false;
  }
  result = NNAdapterCompilation_finish_invoke(compilation_);
  if (result != NNADAPTER_NO_ERROR) {
    LOG(WARNING) << "Build model failed(" << result << ") !";
    return false;
  }
  return true;
}

bool Program::BuildAndCacheToFile(
    int block_idx,
    const std::shared_ptr<const cpp::ProgramDesc>& program_desc,
    Scope* exec_scope,
    const std::vector<std::string>& input_names,
    std::vector<std::string>* output_names,
    const std::string& model_cache_token,
    const std::string& model_cache_dir) {
  // Converting the PaddlePaddle operators and variables to the NNAdapter
  // operations and operands for building NNAdapter model(hardware-indepedent)
  CHECK(!model_cache_token.empty());
  int result = NNAdapterModel_create_invoke(&model_);
  subgraph::nnadapter::Converter converter(model_);
  std::unique_ptr<RuntimeProgram> runtime_program(
      new RuntimeProgram(program_desc, exec_scope, block_idx));
  const auto& bridges = subgraph::SubgraphBridgeRegistry::Instance();
  CHECK(runtime_program) << "The runtime program is not initialized!";
  CHECK_GT(runtime_program->instructions(kRootBlockIdx).size(), 0)
      << "No instructions found in the runtime program!";
  const auto& insts = runtime_program->instructions(kRootBlockIdx);
  int status = 0;
  for (auto& inst : insts) {
    auto op = const_cast<OpLite*>(inst.op());
    CHECK(op);
    op->CheckShape();
    op->InferShape();
    std::string op_type = op->op_info()->Type();
    if (!bridges.Exists(op_type, TARGET(kNNAdapter))) {
      return false;
    }
    auto kernel = inst.kernel();
    status |= bridges.Select(op_type, TARGET(kNNAdapter))(
        reinterpret_cast<void*>(&converter),
        op,
        const_cast<KernelBase*>(kernel));
    if (subgraph::CHECK_FAILED(status)) {
      return false;
    }
  }
  // Query and indentify the input and output operands
  std::vector<NNAdapterOperand *> input_operands, output_operands;
  for (size_t i = 0; i < input_names.size(); i++) {
    CHECK(converter.HasOperand(input_names[i]))
        << "No operand found for input '" << input_names[i] << "'!";
    auto operand = converter.GetOperand(input_names[i]);
    input_operands.push_back(operand);
    VLOG(3) << "Found an operand @0x" << string_format("%x", operand)
            << " for input '" << input_names[i] << "'.";
  }
  // Update if exists the useless output variables such as 'XShape' in reshape2
  // and transpose2
  std::vector<std::string> valid_output_names;
  for (size_t i = 0; i < output_names->size(); i++) {
    const auto& output_name = output_names->at(i);
    if (!converter.HasOperand(output_name)) {
      LOG(WARNING) << "No operand found for output '" << output_name << "'!";
      continue;
    }
    auto operand = converter.GetOperand(output_name);
    output_operands.push_back(operand);
    VLOG(3) << "Found an operand @0x" << string_format("%x", operand)
            << " for output '" << output_name << "'.";
    valid_output_names.push_back(output_name);
  }
  CHECK_GT(valid_output_names.size(), 0);
  if (valid_output_names.size() != output_names->size()) {
    *output_names = valid_output_names;
  }
  NNAdapterModel_identifyInputsAndOutputs_invoke(model_,
                                                 input_operands.size(),
                                                 input_operands.data(),
                                                 output_operands.size(),
                                                 output_operands.data());
  result = NNAdapterModel_finish_invoke(model_);
  // Compiling the model to the device-specific binary program
  result = NNAdapterCompilation_create_invoke(model_,
                                              model_cache_token.c_str(),
                                              nullptr,
                                              0,
                                              model_cache_dir.c_str(),
                                              context_,
                                              &compilation_);
  if (result != NNADAPTER_NO_ERROR) {
    NNAdapterModel_destroy_invoke(model_);
    model_ = nullptr;
    LOG(WARNING)
        << "Failed to create a compilation by compiling the source model ("
        << result << ") !";
    return false;
  }
  result = NNAdapterCompilation_finish_invoke(compilation_);
  if (result != NNADAPTER_NO_ERROR) {
    NNAdapterModel_destroy_invoke(model_);
    model_ = nullptr;
    LOG(WARNING) << "Build model failed(" << result << ") !";
    return false;
  }
  return true;
}

bool Program::SetInputsAndOutputs(const std::vector<Tensor*>& input_tensors,
                                  const std::vector<Tensor*>& output_tensors) {
  CHECK(IsValid());
  // Query the information of inputs and outputs
  uint32_t input_count, output_count;
  int result = NNAdapterCompilation_queryInputsAndOutputs_invoke(
      compilation_, &input_count, NULL, &output_count, NULL);
  if (result != NNADAPTER_NO_ERROR) {
    LOG(WARNING) << "Failed to query the count of inputs and outputs from the "
                    "compilation("
                 << result << ") !";
    return false;
  }
  CHECK_EQ(input_count, input_tensors.size());
  CHECK_EQ(output_count, output_tensors.size());
  std::vector<NNAdapterOperandType *> input_types(input_count),
      output_types(output_count);
  result = NNAdapterCompilation_queryInputsAndOutputs_invoke(compilation_,
                                                             &input_count,
                                                             &input_types[0],
                                                             &output_count,
                                                             &output_types[0]);
  if (result != NNADAPTER_NO_ERROR) {
    LOG(WARNING) << "Failed to query the type of inputs and outputs from the "
                    "compilation("
                 << result << ") !";
    return false;
  }
  // Create an execution for executing the compiled device program
  result = NNAdapterExecution_create_invoke(compilation_, &execution_);
  if (result != NNADAPTER_NO_ERROR) {
    LOG(WARNING) << "Create execution failed(" << result << ") !";
    return false;
  }
  // Set the input and output tensors to model and the functions used to access
  // them.
  for (uint32_t i = 0; i < input_count; i++) {
    NNAdapterExecution_setInput_invoke(
        execution_,
        i,
        reinterpret_cast<void*>(input_tensors[i]),
        AccessModelInput);
  }
  for (uint32_t i = 0; i < output_count; i++) {
    NNAdapterExecution_setOutput_invoke(
        execution_,
        i,
        reinterpret_cast<void*>(output_tensors[i]),
        AccessModelOutput);
  }
  return true;
}

bool Program::Execute() {
  CHECK(IsReady());
  auto GetCurrentUS = []() -> double {
    struct timeval time;
    gettimeofday(&time, NULL);
    return 1e+6 * time.tv_sec + time.tv_usec;
  };
  auto start_time = GetCurrentUS();
  int result = NNAdapterExecution_compute_invoke(execution_);
  if (result != NNADAPTER_NO_ERROR) {
    LOG(WARNING) << "Failed to run the execution(" << result << ")!";
    return false;
  }
  VLOG(3) << "Process cost " << GetCurrentUS() - start_time << " us";
  return true;
}

SubgraphEngine::SubgraphEngine(
    KernelContext* ctx,
    int block_idx,
    const std::shared_ptr<const cpp::ProgramDesc>& program_desc,
    Scope* exec_scope,
    const std::vector<std::string>& input_names,
    const std::vector<std::string>& output_names)
    : ctx_(ctx),
      block_idx_(block_idx),
      program_desc_(program_desc),
      exec_scope_(exec_scope),
      input_names_(input_names),
      output_names_(output_names) {
  int result;
  // Obtain the same order every time by sorting the input and output names,
  // because the topological order may be different each time of the partition
  // of the subgraph(but they are equivalent)
  std::stable_sort(input_names_.begin(), input_names_.end());
  std::stable_sort(output_names_.begin(), output_names_.end());
  input_dims_.resize(input_names_.size());
  // Initialize the input and output tensors
  for (size_t i = 0; i < input_names_.size(); i++) {
    input_tensors_.push_back(exec_scope_->FindMutableTensor(input_names_[i]));
  }
  for (size_t i = 0; i < output_names_.size(); i++) {
    output_tensors_.push_back(exec_scope_->FindMutableTensor(output_names_[i]));
  }
  // Get the specified devices and create a context for each device to build or
  // load the device-related program from the model or the cache file/buffer.
  const auto& device_names =
      ctx->As<NNAdapterContext>().NNAdapterDeviceNames(exec_scope_);
  CHECK_GT(device_names.size(), 0) << "No device is specified.";
  for (const auto& device_name : device_names) {
    NNAdapterDevice* device = nullptr;
    result = NNAdapterDevice_acquire_invoke(device_name.c_str(), &device);
    bool found = result == NNADAPTER_NO_ERROR && device != nullptr;
    if (found) {
      const char* name = nullptr;
      NNAdapterDevice_getName_invoke(device, &name);
      const char* vendor = nullptr;
      NNAdapterDevice_getVendor_invoke(device, &vendor);
      NNAdapterDeviceType type = 0;
      NNAdapterDevice_getType_invoke(device, &type);
      int32_t version = 0;
      NNAdapterDevice_getVersion_invoke(device, &version);
      VLOG(3) << "NNAdapter device " << name << ": vendor=" << vendor
              << " type=" << type << " version=" << version;
      devices_.push_back(device);
      // Only support the first found device.
      break;
    }
  }
  CHECK_GT(devices_.size(), 0) << "No device is found.";
  // Get the context properties from the scope
  auto context_properties =
      ctx->As<NNAdapterContext>().NNAdapterContextProperties(exec_scope_);
  VLOG(3) << "NNAdapter context_properties: " << context_properties;
  // Create a context with multiple devices
  NNAdapterContext_create_invoke(
      &devices_[0], devices_.size(), context_properties.c_str(), &context_);
  // Get the model cache dir from the scope
  model_cache_dir_ =
      ctx_->As<NNAdapterContext>().NNAdapterModelCacheDir(exec_scope_);
  VLOG(3) << "NNAdapter model_cache_dir: " << model_cache_dir_;
}

SubgraphEngine::~SubgraphEngine() {
  programs_.clear();
  NNAdapterContext_destroy_invoke(context_);
  for (auto* device : devices_) {
    NNAdapterDevice_release_invoke(device);
  }
}

std::shared_ptr<Program> SubgraphEngine::Build(
    const std::vector<std::vector<int64_t>>& input_dims) {
  // Get the valid device names
  std::vector<std::string> valid_device_names;
  for (auto* device : devices_) {
    const char* name = nullptr;
    NNAdapterDevice_getName_invoke(device, &name);
    valid_device_names.push_back(name);
  }
  auto program = std::make_shared<Program>(context_);
  // Take the model cache buffer from the scope
  std::vector<char> model_cache_buffer;
  auto model_cache_token =
      GenerateModelCacheToken(valid_device_names, input_names_, input_dims);
  ctx_->As<NNAdapterContext>().NNAdapterModelCacheBuffers(
      exec_scope_, model_cache_token, &model_cache_buffer);
  VLOG(3) << "NNAdapter model_cache_buffer size: " << model_cache_buffer.size();
  // Load the compiled device program from the model cache buffer or file
  if (!program->LoadFromCache(
          model_cache_token, &model_cache_buffer, model_cache_dir_)) {
    // Build the model to the compiled device program online and cache it to
    // file
    if (!program->BuildAndCacheToFile(block_idx_,
                                      program_desc_,
                                      exec_scope_,
                                      input_names_,
                                      &output_names_,
                                      model_cache_token,
                                      model_cache_dir_)) {
      return nullptr;
    }
    // Update the output tensors if exists some useless outputs(such as 'XShape'
    // in reshape2 and transpose2)
    if (output_tensors_.size() != output_names_.size()) {
      output_tensors_.clear();
      for (size_t i = 0; i < output_names_.size(); i++) {
        output_tensors_.push_back(
            exec_scope_->FindMutableTensor(output_names_[i]));
      }
    }
  }
  return program->IsValid() &&
                 program->SetInputsAndOutputs(input_tensors_, output_tensors_)
             ? program
             : nullptr;
}

bool SubgraphEngine::Run() {
  // Update the input dimensions
  for (size_t i = 0; i < input_tensors_.size(); i++) {
    input_dims_[i] = input_tensors_[i]->dims().Vectorize();
  }
  // Find, build and execute the device-specific program according to the input
  // dimensions
  std::shared_ptr<Program> program = nullptr;
  if (programs_.count(input_dims_)) {
    program = programs_[input_dims_];
  } else {
    program = Build(input_dims_);
    CHECK(program);
    programs_[input_dims_] = program;
  }
  CHECK(program);
  CHECK(program->IsValid());
  return program->Execute();
}

void SubgraphCompute::PrepareForRun() {
  auto& param = this->Param<param_t>();
  engine_.reset(new SubgraphEngine(ctx_.get(),
                                   param.block_idx,
                                   param.program_desc,
                                   param.exec_scope,
                                   param.input_data_names,
                                   param.output_data_names));
  CHECK(engine_);
}

void SubgraphCompute::Run() {
  CHECK(engine_);
  engine_->Run();
}

}  // namespace nnadapter
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(subgraph,
                     kNNAdapter,
                     kAny,
                     kNCHW,
                     paddle::lite::kernels::nnadapter::SubgraphCompute,
                     def)
    .BindInput("Inputs",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kAny),
                                      DATALAYOUT(kNCHW))})
    .BindOutput("Outputs",
                {LiteType::GetTensorTy(TARGET(kHost),
                                       PRECISION(kAny),
                                       DATALAYOUT(kNCHW))})
    .Finalize();
