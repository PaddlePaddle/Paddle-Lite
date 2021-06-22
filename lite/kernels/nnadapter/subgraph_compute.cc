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

// Generate a simple key to indentify the model by using md5 algrithm based on
// the following information:
// 1. the sorted variable input names
// 2. the shapes of the origin input tensors
// 3. the sorted variable output names
std::string KeyGenerator(const std::vector<std::string>& input_names,
                         const std::vector<std::string>& output_names,
                         const std::vector<std::vector<int64_t>>& input_dims) {
  std::ostringstream os;
  CHECK_EQ(input_names.size(), input_dims.size());
  for (int i = 0; i < input_names.size(); i++) {
    os << input_names[i];
    for (auto dim : input_dims[i]) {
      os << dim;
    }
  }
  for (auto output_name : output_names) {
    os << output_name;
  }
  return MD5(os.str());
}

DeviceProgram::~DeviceProgram() {
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

bool DeviceProgram::LoadFromCache(std::vector<char>* model_cache_buffer,
                                  const std::string& model_cache_dir) {
  return false;
}

bool DeviceProgram::BuildAndCacheToFiles(
    RuntimeProgram* origin_program,
    const std::vector<std::string>& input_names,
    const std::vector<std::string>& output_names,
    const std::string& model_cache_dir) {
  // Converting the PaddlePaddle operators and variables to the NNAdapter
  // operations and operands for building the hardware-indepedent model.
  CHECK(!model_cache_key_.empty());
  int result = NNAdapterModel_create_invoke(&model_);
  subgraph::nnadapter::Converter converter(model_);
  const auto& bridges = subgraph::SubgraphBridgeRegistry::Instance();
  CHECK(origin_program) << "[NNAdapter] The origin program is not initialized!";
  CHECK_GT(origin_program->instructions(kRootBlockIdx).size(), 0)
      << "[NNAdapter] No instructions found in the origin program!";
  const auto& insts = origin_program->instructions(kRootBlockIdx);
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
  for (int i = 0; i < input_names.size(); i++) {
    CHECK(converter.HasOperand(input_names[i]))
        << "No operand found for input '" << input_names[i] << "'!";
    auto operand = converter.GetOperand(input_names[i]);
    input_operands.push_back(operand);
    VLOG(3) << "Found an operand @0x" << string_format("%x", operand)
            << " for input '" << input_names[i] << "'.";
  }
  for (int i = 0; i < output_names.size(); i++) {
    CHECK(converter.HasOperand(output_names[i]))
        << "No operand found for output '" << output_names[i] << "'!";
    auto operand = converter.GetOperand(output_names[i]);
    output_operands.push_back(operand);
    VLOG(3) << "Found an operand @0x" << string_format("%x", operand)
            << " for output '" << output_names[i] << "'.";
  }
  NNAdapterModel_identifyInputsAndOutputs_invoke(model_,
                                                 input_operands.size(),
                                                 &input_operands[0],
                                                 output_operands.size(),
                                                 &output_operands[0]);
  result = NNAdapterModel_finish_invoke(model_);
  // Compiling the model to the hardware-related programs
  result = NNAdapterCompilation_create_invoke(model_,
                                              model_cache_key_.c_str(),
                                              nullptr,
                                              0,
                                              model_cache_dir.c_str(),
                                              context_,
                                              &compilation_);
  if (result != NNADAPTER_NO_ERROR) {
    NNAdapterModel_destroy_invoke(model_);
    model_ = nullptr;
    LOG(WARNING) << "Create a compilation for building model failed(" << result
                 << ") !";
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

bool DeviceProgram::SetInputsAndOutputs(std::vector<Tensor*>* origin_itensors,
                                        std::vector<Tensor*>* origin_otensors) {
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
  CHECK_EQ(input_count, origin_itensors->size());
  CHECK_EQ(output_count, origin_otensors->size());
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
  // Set the real dimensions and buffer of the inputs and outputs
  for (size_t i = 0; i < origin_itensors->size(); i++) {
    auto dimensions =
        subgraph::nnadapter::ConvertDimensions(origin_itensors->at(i)->dims());
    NNAdapterExecution_setInput_invoke(execution_,
                                       i,
                                       &dimensions[0],
                                       dimensions.size(),
                                       origin_itensors->at(i)->raw_data(),
                                       origin_itensors->at(i)->memory_size());
  }
  for (size_t i = 0; i < origin_otensors->size(); i++) {
    auto precision =
        subgraph::nnadapter::ConvertPrecision(output_types[i]->precision);
    auto dimensions = subgraph::nnadapter::ConvertDimensions(
        output_types[i]->dimensions, output_types[i]->dimension_count);
    origin_otensors->at(i)->Resize(dimensions);
#define TENSOR_MUTABLE_DATA(ptype, dtype)          \
  case PRECISION(ptype):                           \
    origin_otensors->at(i)->mutable_data<dtype>(); \
    break;
    switch (precision) {
      TENSOR_MUTABLE_DATA(kInt8, int8_t)
      TENSOR_MUTABLE_DATA(kInt32, int32_t)
      TENSOR_MUTABLE_DATA(kFloat, float)
      default:
        LOG(ERROR) << "Failed to mutable data for the precsion type("
                   << PrecisionToStr(precision) << ") of output[" << i << "]!";
        break;
    }
#undef TENSOR_MUTABLE_DATA
    NNAdapterExecution_setOutput_invoke(execution_,
                                        i,
                                        output_types[i]->dimensions,
                                        output_types[i]->dimension_count,
                                        origin_otensors->at(i)->raw_data(),
                                        origin_otensors->at(i)->memory_size());
  }
  return true;
}

bool DeviceProgram::Execute() {
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
    : subgraph::SubgraphEngineBase(
          ctx, block_idx, program_desc, exec_scope, input_names, output_names) {
  int result;
  // Get the device names from the scope
  auto device_names = ctx->As<NNAdapterContext>().NNAdapterDevices(exec_scope);
  CHECK_GT(device_names.size(), 0) << "No device is specified.";
  // Get the specified devices and create a context for each device to build or
  // load the device-related program from the model or the cache files/buffers.
  for (auto& device_name : device_names) {
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
      VLOG(3) << "nnadapter device " << name << ": vendor=" << vendor
              << " type=" << type << " version=" << version;
      devices_.push_back(device);
      // Only support the first found device.
      break;
    }
  }
  CHECK_GT(devices_.size(), 0) << "No device is found.";
  // Create a context with multiple devices
  NNAdapterContext_create_invoke(&devices_[0], devices_.size(), &context_);
  // Get the model cache dir from the scope
  model_cache_dir_ =
      ctx_->As<NNAdapterContext>().NNAdapterModelCacheDir(exec_scope_);
  VLOG(3) << "nnadapter model_cache_dir: " << model_cache_dir_;
}

SubgraphEngine::~SubgraphEngine() {
  NNAdapterContext_destroy_invoke(context_);
  for (auto* device : devices_) {
    NNAdapterDevice_release_invoke(device);
  }
}

bool SubgraphEngine::BuildDeviceProgram() {
  // Check if the compiled device program exists
  if (!device_programs_.count(origin_idims_)) {
    std::string model_cache_key =
        KeyGenerator(input_names_, output_names_, origin_idims_);
    auto device_program =
        std::make_shared<DeviceProgram>(model_cache_key, context_);
    // Load the compiled device program from the buffers which are stored as the
    // tensors in the scope
    std::vector<char> model_cache_buffer;
    ctx_->As<NNAdapterContext>().NNAdapterModelCacheBuffers(
        exec_scope_, model_cache_key, &model_cache_buffer);
    VLOG(3) << "nnadapter model_cache_buffer size: "
            << model_cache_buffer.size();
    // Load if the compiled device program exists
    if (!device_program->LoadFromCache(&model_cache_buffer, model_cache_dir_)) {
      if (!origin_program_) {
        BuildOriginProgram();
      }
      CHECK(origin_program_) << "The origin program is not initialized!";
      CHECK_GT(origin_program_->instructions().size(), 0)
          << "No instructions found in the origin program!";
      // Build the model online and cache to the files
      if (!device_program->BuildAndCacheToFiles(origin_program_.get(),
                                                input_names_,
                                                output_names_,
                                                model_cache_dir_)) {
        return false;
      }
    }
    if (!device_program->IsValid()) {
      return false;
    }
    device_programs_[origin_idims_] = device_program;
  }
  auto device_program = device_programs_[origin_idims_];
  CHECK(device_program && device_program->IsValid());
  return device_program->SetInputsAndOutputs(&origin_itensors_,
                                             &origin_otensors_);
}

bool SubgraphEngine::LaunchDeviceProgram() {
  // Fallback to launch the origin program if the device program is not found or
  // initialized.
  if (!device_programs_.count(origin_idims_)) {
    return LaunchOriginProgram();
  }
  auto device_program = device_programs_[origin_idims_];
  if (!device_program->IsValid()) {
    return LaunchOriginProgram();
  }
  return device_program->Execute();
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
