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

#include "lite/core/arena/framework.h"
#include <set>
#include "lite/core/context.h"
#include "lite/operators/subgraph_op.h"

namespace paddle {
namespace lite {
namespace arena {

void TestCase::CreateInstruction() {
  std::shared_ptr<lite::OpLite> op = nullptr;
  static const std::set<TargetType> subgraph_op_supported_targets(
      {TARGET(kNPU), TARGET(kXPU), TARGET(kHuaweiAscendNPU)});
  bool enable_subgraph_op = subgraph_op_supported_targets.find(place_.target) !=
                            subgraph_op_supported_targets.end();
#if defined(LITE_WITH_XPU) && !defined(LITE_WITH_XTCL)
  enable_subgraph_op = false;  // Use XPU kernel directly if XTCL is disabled.
#endif
  if (enable_subgraph_op) {
    // Create a new block desc to wrap the original op desc
    auto sub_program_desc = std::make_shared<cpp::ProgramDesc>();
    int sub_block_idx = 0;
    auto sub_block_desc = sub_program_desc->AddBlock<cpp::BlockDesc>();
    sub_block_desc->ClearOps();
    sub_block_desc->ClearVars();
    auto sub_op_desc = sub_block_desc->AddOp<cpp::OpDesc>();
    *sub_op_desc = *op_desc_;
    // Add the block desc into the subgraph op which used to replace the
    // original op
    op_desc_.reset(new cpp::OpDesc());
    op_desc_->SetType("subgraph");
    op_desc_->SetAttr<int32_t>("sub_block", sub_block_idx);
    auto in_names = sub_op_desc->input_vars();
    auto out_names = sub_op_desc->output_vars();
    op_desc_->SetInput("Inputs", in_names);
    op_desc_->SetOutput("Outputs", out_names);
    // filter only data op (not const op by persisiable)
    std::vector<std::string> in_data_names;
    for (auto name : in_names) {
      if (!(inst_scope_->FindTensor(name)->persistable())) {
        in_data_names.push_back(name);
      }
    }
    op_desc_->SetAttr<std::vector<std::string>>("input_data_names",
                                                in_data_names);
    op_desc_->SetAttr<std::vector<std::string>>("output_data_names", out_names);
    op = LiteOpRegistry::Global().Create(op_desc().Type());
    static_cast<operators::SubgraphOp*>(op.get())->SetProgramDesc(
        sub_program_desc);
  } else {
    op = LiteOpRegistry::Global().Create(op_desc().Type());
  }
  CHECK(op) << "no op for " << op_desc().Type();
  op->Attach(*op_desc_, inst_scope_.get());
  auto kernels = op->CreateKernels({place_});
  // filter out the target kernel
  CHECK(!kernels.empty()) << "No kernel found for place "
                          << place_.DebugString();
  auto it = std::find_if(
      kernels.begin(), kernels.end(), [&](std::unique_ptr<KernelBase>& k) {
        return k->alias() == alias_;
      });
  CHECK(it != kernels.end()) << "failed to create the kernel in "
                             << place_.DebugString()
                             << " with alias: " << alias_;
  // reset final place
  place_ = (*it)->place();
  // prepare context
  (*it)->SetContext(std::move(ctx_));
  instruction_.reset(new Instruction(op, std::move(*it)));
#ifdef LITE_WITH_PROFILE
  instruction_->set_profiler(new profile::Profiler());
#endif
}

void TestCase::PrepareInputsForInstruction() {
  for (auto& arg : op_desc().InputArgumentNames()) {
    for (auto& var : op_desc().Input(arg)) {
      const auto* type = instruction_->kernel()->GetInputDeclType(arg);
      CHECK(base_scope_->FindVar(var));
      /// Create a tensor or tensor_array in the instruction's scope,
      /// alloc memory and then copy data there.
      if (type->IsTensor() &&
          !TargetCompatibleTo(*Type::GetTensorTy(TARGET(kHost)), *type)) {
        const auto* base_tensor = base_scope_->FindTensor(var);
        auto* inst_tensor = inst_scope_->FindMutableTensor(var);
        CHECK(!base_tensor->dims().empty())
            << "The dims of input tensor is empty yet";
#ifdef LITE_WITH_OPENCL
        input_cpu_tensor.Resize(base_tensor->dims());
        base_tensor->raw_data();
        base_tensor->memory_size();
        input_cpu_tensor.raw_data();
        float* input_cpu_data = input_cpu_tensor.mutable_data<float>();
        memcpy(input_cpu_data,
               base_tensor->raw_data(),
               base_tensor->numel() * sizeof(float));
        const DDim& input_image_dims =
            converter.InitImageDimInfoWith(base_tensor->dims());
        input_image_cpu_tensor.Resize(
            {1, input_image_dims[0], input_image_dims[1], 4});
        uint16_t* input_image_cpu_data =
            input_image_cpu_tensor.mutable_data<uint16_t>();
        converter.NCHWToImage(
            input_cpu_data, input_image_cpu_data, base_tensor->dims());
        inst_tensor->mutable_data<half_t, cl::Image2D>(
            input_image_dims[0], input_image_dims[1], input_image_cpu_data);
#else
        TargetCopy(type->target(),
                   inst_tensor->mutable_data(type->target(),
                                             base_tensor->memory_size()),
                   base_tensor->raw_data(),
                   base_tensor->memory_size());
#endif
      } else if (type->IsTensorList() &&
                 !TargetCompatibleTo(*Type::GetTensorListTy(TARGET(kHost)),
                                     *type)) {
        const auto* base_tensor_list = base_scope_->FindTensorList(var);
        auto* inst_tensor_list = inst_scope_->FindMutableTensorList(var);
        CHECK_EQ(base_tensor_list->size(), inst_tensor_list->size());
        for (size_t i = 0; i < base_tensor_list->size(); i++) {
          CHECK(!base_tensor_list->at(i).dims().empty())
              << "The dims of input tensor[" << i << "] is empty yet";
          TargetCopy(type->target(),
                     inst_tensor_list->at(i).mutable_data(
                         type->target(), base_tensor_list->at(i).memory_size()),
                     inst_tensor_list->at(i).raw_data(),
                     inst_tensor_list->at(i).memory_size());
        }
      }
    }
  }
}

template <typename T>
bool TestCase::CheckTensorPrecision(const Tensor* inst_tensor,
                                    const Tensor* base_tensor,
                                    float abs_error) {
  CHECK(inst_tensor);
  CHECK(base_tensor);

  CHECK(ShapeEquals(inst_tensor->dims(), base_tensor->dims()));

  CHECK(inst_tensor->lod() == base_tensor->lod()) << "lod not match";

  // The baseline should output in host devices.
  CHECK(base_tensor->target() == TARGET(kHost) ||
        base_tensor->target() == TARGET(kX86) ||
        base_tensor->target() == TARGET(kARM));
  const T* inst_data{};
  Tensor inst_host_tensor;
  inst_host_tensor.Resize(inst_tensor->dims());
  switch (inst_tensor->target()) {
    case TARGET(kX86):
    case TARGET(kHost):
    case TARGET(kARM):
      inst_data = static_cast<const T*>(inst_tensor->raw_data());
      break;
#ifdef LITE_WITH_XPU
    case TARGET(kXPU): {
      CopySync<TARGET(kXPU)>(inst_host_tensor.mutable_data<T>(),
                             inst_tensor->raw_data(),
                             sizeof(T) * inst_tensor->dims().production(),
                             IoDirection::DtoH);
      inst_data = inst_host_tensor.data<T>();
      break;
    }
#endif
#ifdef LITE_WITH_OPENCL
    case TARGET(kOpenCL): {
      CLRuntime::Global()->command_queue().finish();
      const DDim& out_image_shape =
          converter.InitImageDimInfoWith(inst_tensor->dims());
      auto out_image_width = out_image_shape[0];
      auto out_image_height = out_image_shape[1];
      half_t* out_image_data = new half_t[out_image_shape.production() * 4];
      auto* out_image = inst_tensor->data<half_t, cl::Image2D>();
      TargetWrapperCL::ImgcpySync(out_image_data,
                                  out_image,
                                  out_image_width,
                                  out_image_height,
                                  0,
                                  0,
                                  IoDirection::DtoH);

      float* out_data = new float[out_image_shape.production() * 4];
      converter.ImageToNCHW(out_image_data,
                            inst_host_tensor.mutable_data<float>(),
                            out_image_shape,
                            inst_tensor->dims());
      inst_data = inst_host_tensor.data<T>();
      break;
    }
#endif
    default:
      // Before compare, need to copy data from `target` device to host.
      LOG(FATAL) << "Not supported";
  }

  CHECK(inst_data);

  const T* base_data = static_cast<const T*>(base_tensor->raw_data());

  bool success = true;
  for (int i = 0; i < inst_tensor->dims().production(); i++) {
    EXPECT_NEAR(inst_data[i], base_data[i], abs_error);
    if (fabsf(inst_data[i] - base_data[i]) > abs_error) {
      success = false;
    }
  }
  return success;
}

bool TestCase::CheckPrecision(const Tensor* inst_tensor,
                              const Tensor* base_tensor,
                              float abs_error,
                              PrecisionType precision_type) {
  PrecisionType precision_type_t = precision_type;
  if (precision_type == PRECISION(kAny)) {
    precision_type_t = base_tensor->precision();
  }
#ifdef LITE_WITH_OPENCL
  precision_type_t = base_tensor->precision();
#endif
  CHECK(precision_type_t == base_tensor->precision())
      << "arg precision type and base tensor precision type are not matched! "
         "arg precision type is: "
      << PrecisionToStr(precision_type) << ", base tensor precision type is: "
      << PrecisionToStr(base_tensor->precision());
#ifdef LITE_WITH_OPENCL

#else
  CHECK(inst_tensor->precision() == base_tensor->precision())
      << "real tensor precision type and base tensor precision type are not "
         "matched! real tensor precision type is: "
      << PrecisionToStr(inst_tensor->precision())
      << ", base tensor precision type is: "
      << PrecisionToStr(base_tensor->precision());
#endif
  switch (precision_type_t) {
    case PRECISION(kFloat):
      return CheckTensorPrecision<float>(inst_tensor, base_tensor, abs_error);
    case PRECISION(kInt8):
      return CheckTensorPrecision<int8_t>(inst_tensor, base_tensor, abs_error);
    case PRECISION(kInt32):
      return CheckTensorPrecision<int32_t>(inst_tensor, base_tensor, abs_error);
    case PRECISION(kInt64):
      return CheckTensorPrecision<int64_t>(inst_tensor, base_tensor, abs_error);
    case PRECISION(kBool):
      return CheckTensorPrecision<bool>(inst_tensor, base_tensor, abs_error);
    default:
      LOG(FATAL) << "not support type: " << PrecisionToStr(precision_type);
      return false;
  }
}

bool TestCase::CheckPrecision(const std::string& var_name,
                              float abs_error,
                              PrecisionType precision_type) {
  bool success = true;
  if (inst_scope_->FindVar(var_name)->IsType<Tensor>()) {
    auto inst_tensor = inst_scope_->FindTensor(var_name);
    auto base_tensor = base_scope_->FindTensor(var_name);
    success =
        success &&
        CheckPrecision(inst_tensor, base_tensor, abs_error, precision_type);
  } else if (inst_scope_->FindVar(var_name)->IsType<std::vector<Tensor>>()) {
    auto inst_tensor_list = inst_scope_->FindMutableTensorList(var_name);
    auto base_tensor_list = base_scope_->FindMutableTensorList(var_name);
    CHECK_EQ(inst_tensor_list->size(), base_tensor_list->size());
    for (size_t i = 0; i < inst_tensor_list->size(); i++) {
      Tensor* inst_tensor = &(inst_tensor_list->at(i));
      Tensor* base_tensor = &(base_tensor_list->at(i));
      if (inst_tensor->dims().size() == 0 && base_tensor->dims().size() == 0) {
        continue;
      }
      success =
          success &&
          CheckPrecision(inst_tensor, base_tensor, abs_error, precision_type);
    }
  } else {
    LOG(FATAL) << "unsupported var type";
  }
  return success;
}

}  // namespace arena
}  // namespace lite
}  // namespace paddle
