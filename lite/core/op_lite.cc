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

#include "lite/core/op_lite.h"
#include <list>
#include <set>
#include <utility>
#include <vector>
#include "lite/core/op_registry.h"
#include "lite/utils/string.h"

namespace paddle {
namespace lite {

bool OpLite::InferShape() {
  // if input_tensor_ptrs and output_tensor_ptrs are overloaded in param_
  // InferShapeByMemoryInternal will be applied.
  if (op_param_ && op_param_->input_tensor_ptrs() &&
      op_param_->output_tensor_ptrs()) {
    return this->InferShapeWithCache();
  } else {
    return this->InferShapeImpl();
  }
}
bool OpLite::InferShapeWithCache() {
  // 1. Get vector of current input tensors
  auto *current_inputs = op_param_->input_tensor_ptrs();
  // 2. Get hash value of current inputs shape and lod
  bool use_cache = true;
  if (last_input_shapes.size() == current_inputs->size()) {
    for (size_t i = 0; i < current_inputs->size(); i++) {
      if (last_input_shapes[i] != current_inputs->at(i)->dims() ||
          last_input_lods[i] != current_inputs->at(i)->lod()) {
        use_cache = false;
        break;
      }
    }
  } else {
    use_cache = false;
  }

  // 3. infer shapes of output tensors
  if (use_cache) {
    // if current hash value is consistent with io_shape_lod_hash_,
    // previous outputs shape and lod are reused.
    auto *current_outputs = op_param_->output_tensor_ptrs();
    for (size_t i = 0; i < current_outputs->size(); i++) {
      current_outputs->at(i)->Resize(last_output_shapes[i]);
      current_outputs->at(i)->set_lod(last_output_lods[i]);
    }
  } else {
    // otherwise, current hash value is changed, InferShapeImpl will apply.
    this->InferShapeImpl();
    auto *current_outputs = op_param_->output_tensor_ptrs();
    last_output_shapes.clear();
    last_output_lods.clear();
    for (size_t i = 0; i < current_outputs->size(); i++) {
      last_output_shapes.push_back(current_outputs->at(i)->dims());
      last_output_lods.push_back(current_outputs->at(i)->lod());
    }
    last_input_shapes.clear();
    last_input_lods.clear();
    for (size_t i = 0; i < current_inputs->size(); i++) {
      last_input_shapes.push_back(current_inputs->at(i)->dims());
      last_input_lods.push_back(current_inputs->at(i)->lod());
    }
  }
  return true;
}

std::vector<std::unique_ptr<KernelBase>> OpLite::CreateKernels(
    const std::vector<Place> &places, const std::string &kernel_type) {
  std::vector<std::unique_ptr<KernelBase>> kernels;
  CHECK(!op_type_.empty()) << "op_type_ should be set first";

  auto pick_kernel = [&](const Place &place) {
    auto ks = KernelRegistry::Global().Create(
        op_type_, place.target, place.precision, place.layout);
    VLOG(5) << "pick kernel for " << op_info()->Type() << " "
            << place.DebugString() << " get " << ks.size() << " kernels";
    for (auto &&it : ks) {
      AttachKernel(it.get());
      kernels.emplace_back(std::move(it));
    }
  };

  if (!kernel_type.empty()) {
    Place place;
    std::string op_type, alias;
    KernelBase::ParseKernelType(kernel_type, &op_type, &alias, &place);
    pick_kernel(place);
    CHECK(!kernels.empty()) << "no kernel for kernel type " << kernel_type;
    return kernels;
  }

  std::set<Place> expanded_places(places.begin(), places.end());
  for (auto &place : places) {
    // Pick kernels those support any Precision and any DataLayout, For example:
    // kARM,kFloat,kNCHW -> kARM,kFloat,kAny; kARM,kAny,kNCHW; kARM,kAny,kAny
    expanded_places.insert(
        Place(place.target, place.precision, DATALAYOUT(kAny)));
    expanded_places.insert(Place(place.target, PRECISION(kAny), place.layout));
    expanded_places.insert(
        Place(place.target, PRECISION(kAny), DATALAYOUT(kAny)));
  }

  std::set<TargetType> targets;
  for (auto place : expanded_places) {
    pick_kernel(place);
    targets.insert(place.target);
  }

  VLOG(5) << "op " << op_type_ << " get " << kernels.size() << " kernels";
  return kernels;
}

bool OpLite::Run() {
  CHECK(kernel_);
  SyncInputEvents();

  kernel_->Launch();

  RecordOutputEvents();
  return true;
}

bool OpLite::Attach(const cpp::OpDesc &opdesc, lite::Scope *scope) {
  // valid_places_.clear();
  CHECK(scope != nullptr);
  // CHECK(!op_info_.get());
  scope_ = scope;
  op_info_.reset(
      new OpInfo(opdesc));  // Force clean the out-of-date infomation.
  return AttachImpl(*op_info(), scope);
}

const Tensor *OpLite::GetTensor(lite::Scope *scope,
                                const std::string &name) const {
  auto *var = scope->FindVar(name);
  CHECK(var) << "no variable called " << name << " found";
  return &var->Get<lite::Tensor>();
}

Tensor *OpLite::GetMutableTensor(lite::Scope *scope,
                                 const std::string &name) const {
  auto *var = scope->FindVar(name);
  CHECK(var) << "no variable called " << name << " found";
  return var->GetMutable<lite::Tensor>();
}

void OpLite::AttachInput(const cpp::OpDesc &op_desc,
                         lite::Scope *scope,
                         const std::string &input_name,
                         bool is_dispensable,
                         lite::Tensor **input_var) {
  bool is_have_input =
      op_desc.HasInput(input_name) && op_desc.Input(input_name).size() > 0;
  CHECK(is_dispensable || is_have_input);
  if (is_have_input) {
    std::string input_var_name = op_desc.Input(input_name).front();
    *input_var = scope->FindVar(input_var_name)->GetMutable<lite::Tensor>();
  }
}

void OpLite::AttachOutput(const cpp::OpDesc &op_desc,
                          lite::Scope *scope,
                          const std::string &output_name,
                          bool is_dispensable,
                          lite::Tensor **output_var) {
  bool is_have_output =
      op_desc.HasOutput(output_name) && op_desc.Output(output_name).size() > 0;
  CHECK(is_dispensable || is_have_output);
  if (is_have_output) {
    std::string output_var_name = op_desc.Output(output_name).front();
    *output_var = scope->FindVar(output_var_name)->GetMutable<lite::Tensor>();
  }
}

bool OpInfo::GetInputArgname(const std::string &value_name,
                             std::string *out) const {
  for (auto &item : inputs()) {
    auto it = std::find(item.second.begin(), item.second.end(), value_name);
    if (it != item.second.end()) {
      *out = item.first;
      return true;
    }
  }
  return false;
}

bool OpInfo::GetOutputArgname(const std::string &value_name,
                              std::string *out) const {
  for (auto &item : outputs()) {
    auto it = std::find(item.second.begin(), item.second.end(), value_name);
    if (it != item.second.end()) {
      *out = item.first;
      return true;
    }
  }
  return false;
}

bool OpInfo::GetInputIndex(const std::string &input_name, int *out) const {
  for (auto &item : inputs()) {
    auto it = std::find(item.second.begin(), item.second.end(), input_name);
    if (it != item.second.end()) {
      *out = it - item.second.begin();
      return true;
    }
  }
  return false;
}

bool OpInfo::GetOutputIndex(const std::string &output_name, int *out) const {
  for (auto &item : outputs()) {
    auto it = std::find(item.second.begin(), item.second.end(), output_name);
    if (it != item.second.end()) {
      *out = it - item.second.begin();
      return true;
    }
  }
  return false;
}

bool OpInfo::HasInputScale(const std::string &name, bool is_scale_name) const {
  bool res = false;
  if (is_scale_name) {
    res = HasAttr(name);
  } else {
    std::string argname;
    int index;
    if (GetInputArgname(name, &argname) && GetInputIndex(name, &index)) {
      res = HasAttr(argname + to_string(index) + "_scale");
    }
  }
  return res;
}

bool OpInfo::HasOutputScale(const std::string &name, bool is_scale_name) const {
  bool res = false;
  if (is_scale_name) {
    res = HasAttr(name);
  } else {
    std::string argname;
    int index;
    if (GetOutputArgname(name, &argname) && GetOutputIndex(name, &index)) {
      res = HasAttr(argname + to_string(index) + "_scale");
    }
  }
  return res;
}

void OpInfo::SetInputScale(const std::string &name,
                           const std::vector<float> &scale_value,
                           bool is_scale_name) {
  std::string scale_name;
  if (is_scale_name) {
    scale_name = name;
  } else {
    std::string argname;
    int index;
    CHECK(GetInputArgname(name, &argname));
    CHECK(GetInputIndex(name, &index));
    CHECK(scale_value.size() > 0)
        << "Error in SetInputScale: the scales should not be empty";
    scale_name = argname + to_string(index) + "_scale";
  }
  SetAttr<std::vector<float>>(scale_name, scale_value);
}

void OpInfo::SetOutputScale(const std::string &name,
                            const std::vector<float> &scale_value,
                            bool is_scale_name) {
  std::string scale_name;
  if (is_scale_name) {
    scale_name = name;
  } else {
    std::string argname;
    int index;
    CHECK(GetOutputArgname(name, &argname));
    CHECK(GetOutputIndex(name, &index));
    CHECK(scale_value.size() > 0)
        << "Error in SetOutputScale: the scales should not be empty";
    scale_name = argname + to_string(index) + "_scale";
  }
  SetAttr<std::vector<float>>(scale_name, scale_value);
}

std::vector<float> OpInfo::GetInputScale(const std::string &name,
                                         bool is_scale_name) const {
  std::string scale_name;
  if (is_scale_name) {
    scale_name = name;
  } else {
    std::string argname;
    int index;
    CHECK(GetInputArgname(name, &argname));
    CHECK(GetInputIndex(name, &index));
    scale_name = argname + to_string(index) + "_scale";
  }
  return GetAttr<std::vector<float>>(scale_name);
}

std::vector<float> OpInfo::GetOutputScale(const std::string &name,
                                          bool is_scale_name) const {
  std::string scale_name;
  if (is_scale_name) {
    scale_name = name;
  } else {
    std::string argname;
    int index;
    CHECK(GetOutputArgname(name, &argname));
    CHECK(GetOutputIndex(name, &index));
    scale_name = argname + to_string(index) + "_scale";
  }
  return GetAttr<std::vector<float>>(scale_name);
}

}  // namespace lite
}  // namespace paddle
