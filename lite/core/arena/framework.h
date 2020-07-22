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
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include "lite/core/op_registry.h"
#include "lite/core/program.h"
#include "lite/core/scope.h"
#include "lite/core/types.h"
#include "lite/model_parser/cpp/op_desc.h"

namespace paddle {
namespace lite {
namespace arena {

/*
 * Init data and prepare the op.
 */
class TestCase {
 public:
  explicit TestCase(const Place& place, const std::string& alias)
      : place_(place), scope_(new Scope), alias_(alias) {
    ctx_ = ContextScheduler::Global().NewContext(place_.target);
  }
  virtual ~TestCase() {}

  void Prepare() {
    PrepareScopes();
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

  Scope& scope() { return *scope_; }

  Scope* baseline_scope() { return base_scope_; }
  Scope* inst_scope() { return inst_scope_; }

 protected:
  // Prepare inputs in scope() for Tester.
  virtual void PrepareData() = 0;

  /// Prepare a tensor in host. The tensors will be created in scope_.
  /// Need to specify the targets other than X86 or ARM.
  template <typename T>
  void SetCommonTensor(const std::string& var_name,
                       const DDim& ddim,
                       const T* data,
                       const LoD& lod = {},
                       bool is_persistable = false) {
    auto* tensor = scope_->NewTensor(var_name);
    tensor->Resize(ddim);
    auto* d = tensor->mutable_data<T>();
    memcpy(d, data, ddim.production() * sizeof(T));

    // set lod
    if (!lod.empty()) *tensor->mutable_lod() = lod;
    // set persistable
    tensor->set_persistable(is_persistable);
  }

  /// Prepare a tensor_array in host. The tensors will be created in scope_.
  /// Need to specify the targets other than X86 or ARM.
  template <typename T>
  void SetCommonTensorList(const std::string& var_name,
                           const std::vector<DDim>& array_tensor_dims,
                           const std::vector<std::vector<T>>& datas,
                           const std::vector<LoD>& lods = {}) {
    CHECK_EQ(array_tensor_dims.size(), datas.size());
    if (!lods.empty()) {
      CHECK_EQ(array_tensor_dims.size(), lods.size());
    }

    auto* tensor_array =
        scope_->Var(var_name)->GetMutable<std::vector<Tensor>>();
    for (int i = 0; i < array_tensor_dims.size(); i++) {
      Tensor tmp;
      tmp.Resize(array_tensor_dims[i]);
      auto* tmp_data = tmp.mutable_data<T>();
      memcpy(tmp_data,
             datas[i].data(),
             array_tensor_dims[i].production() * sizeof(T));
      if (!lods.empty()) {
        tmp.set_lod(lods[i]);
      }
      tensor_array->push_back(tmp);
    }
  }

  // Prepare for the operator.
  virtual void PrepareOpDesc(cpp::OpDesc* op_desc) = 0;

 public:
  const Instruction& instruction() { return *instruction_; }

 private:
  std::unique_ptr<KernelContext> ctx_;
  void CreateInstruction();

  void PrepareScopes() {
    inst_scope_ = &scope_->NewScope();
    base_scope_ = &scope_->NewScope();
  }

  // Check shape
  // TODO(Superjomn) Move this method to utils or DDim?
  bool ShapeEquals(const DDim& a, const DDim& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); i++) {
      if (a[i] != b[i]) return false;
    }
    return true;
  }

  /// Copy the input tensors to target devices needed by the instruction.
  void PrepareInputsForInstruction();

  // Create output tensors and variables.
  void PrepareOutputsForInstruction() {
    for (auto x : op_desc().output_vars()) {
      inst_scope_->NewTensor(x);
      base_scope_->NewTensor(x);
    }
  }

 private:
  Place place_;
  std::shared_ptr<Scope> scope_;
  std::string alias_;
  // The workspace for the Instruction.
  Scope* inst_scope_{};
  // The workspace for the baseline implementation.
  Scope* base_scope_{};
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
