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
#include "lite/core/context.h"
#include "lite/operators/subgraph_op.h"

namespace paddle {
namespace lite {
namespace arena {

void TestCase::CreateInstruction() {
  std::shared_ptr<lite::OpLite> op = nullptr;
  if (place_.target == TARGET(kNPU) || place_.target == TARGET(kXPU)) {
    // Create a new block desc to wrap the original op desc
    int sub_block_idx = 0;
    auto sub_block_desc = new cpp::BlockDesc();
    sub_block_desc->ClearOps();
    sub_block_desc->ClearVars();
    auto sub_block_op_desc = sub_block_desc->AddOp<cpp::OpDesc>();
    *sub_block_op_desc = *op_desc_;
    // Add the block desc into the subgraph op which used to replace the
    // original op
    op_desc_.reset(new cpp::OpDesc());
    op_desc_->SetType("subgraph");
    op_desc_->SetAttr<int32_t>("sub_block", sub_block_idx);
    auto in_names = sub_block_op_desc->input_vars();
    auto out_names = sub_block_op_desc->output_vars();
    op_desc_->SetInput("Inputs", in_names);
    op_desc_->SetOutput("Outputs", out_names);
    op_desc_->SetAttr<std::vector<std::string>>("input_data_names", in_names);
    op_desc_->SetAttr<std::vector<std::string>>("output_data_names", out_names);
    op = LiteOpRegistry::Global().Create(op_desc().Type());
    static_cast<operators::SubgraphOp*>(op.get())->SetSubBlock(sub_block_desc);
  } else {
    op = LiteOpRegistry::Global().Create(op_desc().Type());
  }
  CHECK(op) << "no op for " << op_desc().Type();
  op->Attach(*op_desc_, inst_scope_);
  auto kernels = op->CreateKernels({place_});
  // filter out the target kernel
  CHECK(!kernels.empty()) << "No kernel found for place "
                          << place_.DebugString();
  auto it = std::remove_if(
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
      std::string kernel_key = instruction_->kernel()->key_with_alias();
      const auto* param_type = ParamTypeRegistry::Global().RetrieveInArgument(
          place_, kernel_key, arg);

      const Type* inst_type = nullptr;
      if (param_type->type->IsTensor()) {
        inst_type = Type::GetTensorTy(TARGET(kHost));
      } else if (param_type->type->IsTensorList()) {
        inst_type = Type::GetTensorListTy(TARGET(kHost));
      } else {
        LOG(FATAL) << "unsupported param_type";
      }

      CHECK(scope_->FindVar(var));
      if (!TargetCompatibleTo(*inst_type, *param_type->type)) {
        /// Create a tensor or tensor_array in the instruction's scope,
        /// alloc memory and then copy data there.
        if (param_type->type->IsTensor()) {
          const auto* shared_tensor = scope_->FindTensor(var);
          auto* target_tensor = inst_scope_->NewTensor(var);
          CHECK(!shared_tensor->dims().empty()) << "shared_tensor is empty yet";
          target_tensor->Resize(shared_tensor->dims());
          TargetCopy(param_type->type->target(),
                     target_tensor->mutable_data(param_type->type->target(),
                                                 shared_tensor->memory_size()),
                     shared_tensor->raw_data(),
                     shared_tensor->memory_size());
        } else if (param_type->type->IsTensorList()) {
          const auto* shared_tensor_array =
              scope_->FindVar(var)->GetMutable<std::vector<Tensor>>();
          auto* target_tensor_array =
              inst_scope_->Var(var)->GetMutable<std::vector<Tensor>>();
          CHECK(!shared_tensor_array->empty())
              << "shared_tensor_array is empty yet";
          target_tensor_array->resize(shared_tensor_array->size());
          for (size_t i = 0; i < shared_tensor_array->size(); i++) {
            target_tensor_array->at(i).Resize(
                shared_tensor_array->at(i).dims());
            TargetCopy(param_type->type->target(),
                       target_tensor_array->at(i).mutable_data(
                           param_type->type->target(),
                           shared_tensor_array->at(i).memory_size()),
                       shared_tensor_array->at(i).raw_data(),
                       shared_tensor_array->at(i).memory_size());
          }
        } else {
          LOG(FATAL) << "not support";
        }
      }
    }
  }
}

template <typename T>
bool TestCase::CheckTensorPrecision(const Tensor* a_tensor,
                                    const Tensor* b_tensor,
                                    float abs_error) {
  CHECK(a_tensor);
  CHECK(b_tensor);

  CHECK(ShapeEquals(a_tensor->dims(), b_tensor->dims()));

  CHECK(a_tensor->lod() == b_tensor->lod()) << "lod not match";

  // The baseline should output in host devices.
  CHECK(b_tensor->target() == TARGET(kHost) ||
        b_tensor->target() == TARGET(kX86) ||
        b_tensor->target() == TARGET(kARM));

  const T* a_data{};
  switch (a_tensor->target()) {
    case TARGET(kX86):
    case TARGET(kHost):
    case TARGET(kARM):
      a_data = static_cast<const T*>(a_tensor->raw_data());
      break;

    default:
      // Before compare, need to copy data from `target` device to host.
      LOG(FATAL) << "Not supported";
  }

  CHECK(a_data);

  const T* b_data = static_cast<const T*>(b_tensor->raw_data());

  bool success = true;
  for (int i = 0; i < a_tensor->dims().production(); i++) {
    EXPECT_NEAR(a_data[i], b_data[i], abs_error);
    if (fabsf(a_data[i] - b_data[i]) > abs_error) {
      success = false;
    }
  }
  return success;
}

bool TestCase::CheckPrecision(const Tensor* a_tensor,
                              const Tensor* b_tensor,
                              float abs_error,
                              PrecisionType precision_type) {
  PrecisionType precision_type_t = precision_type;
  if (precision_type == PRECISION(kAny)) {
    precision_type_t = b_tensor->precision();
  }
  CHECK(precision_type_t == b_tensor->precision())
      << "arg precision type and base tensor precision type are not matched! "
         "arg precision type is: "
      << PrecisionToStr(precision_type) << ", base tensor precision type is: "
      << PrecisionToStr(b_tensor->precision());
  CHECK(a_tensor->precision() == b_tensor->precision())
      << "real tensor precision type and base tensor precision type are not "
         "matched! real tensor precision type is: "
      << PrecisionToStr(a_tensor->precision())
      << ", base tensor precision type is: "
      << PrecisionToStr(b_tensor->precision());
  switch (precision_type_t) {
    case PRECISION(kFloat):
      return CheckTensorPrecision<float>(a_tensor, b_tensor, abs_error);
    case PRECISION(kInt8):
      return CheckTensorPrecision<int8_t>(a_tensor, b_tensor, abs_error);
    case PRECISION(kInt32):
      return CheckTensorPrecision<int32_t>(a_tensor, b_tensor, abs_error);
    case PRECISION(kInt64):
      return CheckTensorPrecision<int64_t>(a_tensor, b_tensor, abs_error);
    case PRECISION(kBool):
      return CheckTensorPrecision<bool>(a_tensor, b_tensor, abs_error);
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
    auto a_tensor = inst_scope_->FindTensor(var_name);
    auto b_tensor = base_scope_->FindTensor(var_name);
    success = success &&
              CheckPrecision(a_tensor, b_tensor, abs_error, precision_type);
  } else if (inst_scope_->FindVar(var_name)->IsType<std::vector<Tensor>>()) {
    auto a_tensor_array =
        inst_scope_->FindVar(var_name)->GetMutable<std::vector<Tensor>>();
    auto b_tensor_array =
        base_scope_->FindVar(var_name)->GetMutable<std::vector<Tensor>>();
    CHECK_EQ(a_tensor_array->size(), b_tensor_array->size());
    for (size_t i = 0; i < a_tensor_array->size(); i++) {
      Tensor* a_tensor = &(a_tensor_array->at(i));
      Tensor* b_tensor = &(b_tensor_array->at(i));
      if (a_tensor->dims().size() == 0 && b_tensor->dims().size() == 0) {
        continue;
      }
      success = success &&
                CheckPrecision(a_tensor, b_tensor, abs_error, precision_type);
    }
  } else {
    LOG(FATAL) << "unsupported var type";
  }
  return success;
}

TestCase::~TestCase() {
  if (op_desc_->Type() == "subgraph") {
    // Release the subblock desc of Subgraph op
    auto subgraph_op = const_cast<operators::SubgraphOp*>(
        static_cast<const operators::SubgraphOp*>(instruction_->op()));
    CHECK(subgraph_op);
    auto sub_block_desc = subgraph_op->GetSubBlock();
    if (sub_block_desc) {
      delete sub_block_desc;
    }
  }
}

}  // namespace arena
}  // namespace lite
}  // namespace paddle
