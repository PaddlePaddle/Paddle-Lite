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

#pragma once
#include <gtest/gtest.h>
#include <time.h>
#include <algorithm>
#include <chrono>  // NOLINT
#include <cmath>
#include <iomanip>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "lite/core/op_registry.h"
#include "lite/core/program.h"
#include "lite/core/scope.h"
#include "lite/core/types.h"
#include "lite/model_parser/cpp_desc.h"

namespace paddle {
namespace lite {
namespace arena {

/*
 * Init data and prepare the op.
 */
class TestCase {
 public:
  explicit TestCase(const Place& place, const std::string& alias)
      : place_(place),
        alias_(alias),
        inst_scope_(new Scope),
        base_scope_(new Scope) {
    ctx_ = ContextScheduler::Global().NewContext(place_.target);
  }
  virtual ~TestCase() {}

  void Prepare() {
    PrepareData();
    op_desc_.reset(new cpp::OpDesc);
    PrepareOpDesc(op_desc_.get());

    PrepareOutputsForInstruction();
    CreateInstruction();
    PrepareInputsForInstruction();
  }

  /// Run the target instruction, that is run the test operator.
  void RunInstruction() { instruction_->Run(); }

  KernelContext* context() { return ctx_.get(); }

  /// The baseline should be implemented, which acts similar to an operator,
  /// that is take several tensors as input and output several tensors as
  /// output.
  virtual void RunBaseline(Scope* scope) = 0;

  // checkout the precision of the two tensors with type T. b_tensor is baseline
  template <typename T>
  bool CheckTensorPrecision(const Tensor* a_tensor,
                            const Tensor* b_tensor,
                            float abs_error);

  // checkout the precision of the two tensors. b_tensor is baseline
  bool CheckPrecision(const Tensor* a_tensor,
                      const Tensor* b_tensor,
                      float abs_error,
                      PrecisionType precision_type);

  /// Check the precision of the output variables. It will compare the same
  /// tensor (or all tensors of the tensor_array) in two scopes, one of the
  /// instruction execution, and the other for the baseline.
  bool CheckPrecision(const std::string& var_name,
                      float abs_error,
                      PrecisionType precision_type);

  const cpp::OpDesc& op_desc() { return *op_desc_; }

  // Check whether the output tensor is consistent with the output definition in
  // kernel registry.
  void CheckKernelConsistWithDefinition() {}

  Scope* baseline_scope() { return base_scope_.get(); }
  Scope* inst_scope() { return inst_scope_.get(); }

 protected:
  // Prepare inputs in scope() for Tester.
  virtual void PrepareData() = 0;

  /// Prepare a tensor in host. The tensors will be created both in base_scope_
  /// and inst_scope_.
  /// Need to specify the targets other than X86 or ARM.
  template <typename T>
  void SetCommonTensor(const std::string& var_name,
                       const DDim& ddim,
                       const T* data,
                       const LoD& lod = {},
                       bool is_persistable = false) {
    // Create and fill a input tensor with the given data for baseline
    auto* base_tensor = base_scope_->NewTensor(var_name);
    base_tensor->Resize(ddim);
    memcpy(base_tensor->mutable_data<T>(), data, ddim.production() * sizeof(T));

    // set lod
    if (!lod.empty()) *base_tensor->mutable_lod() = lod;
    // set persistable
    base_tensor->set_persistable(is_persistable);

    // Create a copy for instruction
    auto* inst_tensor = inst_scope_->NewTensor(var_name);
    inst_tensor->CopyDataFrom(*base_tensor);
  }

  /// Prepare a tensor_array in host. The tensors will be created in scope_.
  /// Need to specify the targets other than X86 or ARM.
  template <typename T>
  void SetCommonTensorList(const std::string& var_name,
                           const std::vector<DDim>& ddims,
                           const std::vector<std::vector<T>>& datas,
                           const std::vector<LoD>& lods = {}) {
    // Create a tensor array for baseline, and a copy for instruction
    CHECK_EQ(ddims.size(), datas.size());
    if (!lods.empty()) {
      CHECK_EQ(ddims.size(), lods.size());
    }

    auto* base_tensor_list = base_scope_->NewTensorList(var_name);
    auto* inst_tensor_list = inst_scope_->NewTensorList(var_name);
    for (int i = 0; i < ddims.size(); i++) {
      Tensor item;
      item.Resize(ddims[i]);
      memcpy(item.mutable_data<T>(),
             datas[i].data(),
             ddims[i].production() * sizeof(T));
      if (!lods.empty()) {
        item.set_lod(lods[i]);
      }
      base_tensor_list->push_back(item);
      inst_tensor_list->push_back(item);
    }
  }

  // Prepare for the operator.
  virtual void PrepareOpDesc(cpp::OpDesc* op_desc) = 0;

 public:
  const Instruction& instruction() { return *instruction_; }

#ifdef LITE_WITH_OPENCL
  CLImageConverterDefault converter;
  lite::Tensor input_image_cpu_tensor;
  lite::Tensor input_cpu_tensor;
#endif

 private:
  std::unique_ptr<KernelContext> ctx_;
  void CreateInstruction();

  // Check shape
  // TODO(Superjomn) Move this method to utils or DDim?
  bool ShapeEquals(const DDim& a, const DDim& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); i++) {
      if (a[i] != b[i]) return false;
    }
    return true;
  }

  // Copy the host tensors to the device tensors if needed by the instruction.
  void PrepareInputsForInstruction();

  // Create output tensors and variables.
  void PrepareOutputsForInstruction() {
    for (auto x : op_desc().output_vars()) {
      inst_scope_->Var(x);
    }
  }

 private:
  Place place_;
  std::string alias_;
  // The workspace for the Instruction.
  std::shared_ptr<Scope> inst_scope_;
  // The workspace for the baseline implementation.
  std::shared_ptr<Scope> base_scope_;
  std::unique_ptr<cpp::OpDesc> op_desc_;
  std::unique_ptr<Instruction> instruction_;
};

class Arena {
 public:
  Arena(std::unique_ptr<TestCase>&& tester,
        const Place& place,
        float abs_error = 1e-5)
      : tester_(std::move(tester)), place_(place), abs_error_(abs_error) {
    tester_->Prepare();
  }

  bool TestPrecision(const std::vector<std::string>& exclude_outs = {}) {
    tester_->RunBaseline(tester_->baseline_scope());
    tester_->RunInstruction();

    bool success = true;
    for (auto& out : tester_->op_desc().OutputArgumentNames()) {
      for (auto& var : tester_->op_desc().Output(out)) {
        if (std::find(exclude_outs.begin(), exclude_outs.end(), var) !=
            exclude_outs.end()) {
          continue;
        }
        success = success && CompareTensor(out, var);
      }
    }
    LOG(INFO) << "done";
    return success;
  }

  void TestPerformance(int times = 100) {
    auto timer = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < times; i++) {
      tester_->RunInstruction();
    }
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - timer);

    timer = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < times; i++) {
      tester_->RunBaseline(tester_->baseline_scope());
    }
    auto duration_basic = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - timer);
    LOG(INFO) << "average lite duration: " << duration.count() << " ms";
    LOG(INFO) << "average basic duration: " << duration_basic.count() << " ms";
    LOG(INFO) << "speed up ratio: lite_speed / basic_speed: "
              << static_cast<float>(duration_basic.count()) / duration.count();
  }

 private:
  // input_name: X
  bool CompareTensor(const std::string& arg_name, const std::string& var_name) {
    // get tensor type.
    const Type* type =
        tester_->instruction().kernel()->GetOutputDeclType(arg_name);
    auto precision_type = type->precision();
    return tester_->CheckPrecision(var_name, abs_error_, precision_type);
  }

 private:
  std::unique_ptr<TestCase> tester_;
  Place place_;
  float abs_error_;
};

}  // namespace arena
}  // namespace lite
}  // namespace paddle
