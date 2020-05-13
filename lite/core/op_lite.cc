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
#include <limits>
#include <list>
#include <set>
#include <utility>
#include <vector>
#include "lite/core/op_registry.h"

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
    for (int i = 0; i < current_inputs->size(); i++) {
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

size_t KernelGrade(
    const OpInfo &op_info,
    const lite::KernelBase &kernel,
    const std::vector<Place> &places,
    const std::unordered_map<std::string, const Type *> &in_types,
    const std::unordered_map<std::string, const Type *> &out_types,
    const std::vector<std::string> &in_names,
    const std::vector<std::string> &out_names,
    const core::KernelPickFactor &kernel_pick_factors) {
  CHECK_GT(places.size(), 0) << "valid_places is empty.";
  float final_score{-1.};
  Place winner_place{places[0]};
  const int kMax =
      std::numeric_limits<core::KernelPickFactor::value_type>::max();
  size_t place_size = places.size();

  // NOTE: We compare kernel's place with place in valid_places to select the
  // best match place
  //       The place's order in valid_places array decide the user's
  //       preference
  // final_score = weight * socre
  // weight: The weight is compute with (valid_places.size() - i) /
  // valid_places.size() as default.
  //         where i is the place's index in valid_places array.
  // score:  score is the weighted sum of target、percision and layout
  for (size_t i = 0; i < place_size; ++i) {
    const auto &place = places[i];
    float weight = static_cast<float>(place_size - i) / place_size;
    size_t score{};

    VLOG(4) << "Grading kernel: " << kernel.summary() << " for place[" << i
            << "/" << place_size << "]: " << place.DebugString();
    // The more important factor comes first
    if (kernel_pick_factors.IsTargetConsidered() &&
        (place.target == kernel.target() || kernel.target() == TARGET(kAny) ||
         place.target == TARGET(kAny))) {
      score +=
          kMax / static_cast<int>(core::KernelPickFactor::Factor::TargetFirst);
    }
    VLOG(4) << "[score s1]:" << score;
    if (kernel_pick_factors.IsPrecisionConsidered() &&
        (place.precision == kernel.precision() ||
         kernel.precision() == PRECISION(kAny) ||
         place.precision == PRECISION(kAny))) {
      // score skipped, if kernel is int8, but op is not int8
      if (!(kernel.precision() == PRECISION(kInt8) &&
            !op_info.HasAttr("enable_int8"))) {
        score += kMax / static_cast<int>(
                            core::KernelPickFactor::Factor::PrecisionFirst);
      }
    }
    VLOG(4) << "[score s2]:" << score;
    if (kernel_pick_factors.IsDataLayoutConsidered() &&
        (place.layout == kernel.layout() ||
         kernel.layout() == DATALAYOUT(kAny) ||
         place.layout == DATALAYOUT(kAny))) {
      score += kMax / static_cast<int>(
                          core::KernelPickFactor::Factor::DataLayoutFirst);
    }
    VLOG(4) << "[score s3]:" << score;

    // add new rules for precision: When the input types are consistent with
    // kernel's input types  and the output types are consistent with kernel's
    // output types. Select the kernel of the precision. Note that this
    // strategy is not compatible with quantization, so skip quantization op.
    if (!op_info.HasAttr("enable_int8")) {
      bool type_match = true;
      for (size_t i = 0; i < in_names.size(); ++i) {
        std::string arg_name;
        CHECK(op_info.GetInputArgname(in_names[i], &arg_name));
        if (in_types.count(in_names[i])) {
          auto *var_type = in_types.at(in_names[i]);
          auto *arg_type = kernel.GetInputDeclType(arg_name);
          VLOG(4) << "<IN>" << in_names[i] << " var type:" << *var_type
                  << " arg type:" << *arg_type;
          if (!PrecisionCompatibleTo(*var_type, *arg_type)) {
            type_match = false;
            break;
          }
        }
      }
      if (type_match) {
        score *= 2;
      }
      type_match = true;
      for (size_t i = 0; i < out_names.size(); ++i) {
        std::string arg_name;
        CHECK(op_info.GetOutputArgname(out_names[i], &arg_name));
        if (out_types.count(out_names[i])) {
          auto *var_type = out_types.at(out_names[i]);
          auto *arg_type = kernel.GetOutputDeclType(arg_name);
          VLOG(4) << "<OUT>" << out_names[i] << " var type:" << *var_type
                  << " arg type:" << *arg_type;
          if (!PrecisionCompatibleTo(*var_type, *arg_type)) {
            type_match = false;
            break;
          }
        }
      }
      if (type_match) {
        score *= 2;
      }
      VLOG(4) << "[score s4]:" << score;
    }

    if (weight * score > final_score) {
      final_score = weight * score;
      winner_place = place;
    }
  }

  VLOG(4) << "[score(final)]:" << final_score;
  VLOG(4) << "-------- pick summary --------";
  VLOG(4) << " ===> winner_place():" << PrecisionToStr(winner_place.precision)
          << " " << DataLayoutToStr(winner_place.layout) << " "
          << TargetToStr(winner_place.target);
  VLOG(4) << " ===> kernel.place():" << PrecisionToStr(kernel.place().precision)
          << " " << DataLayoutToStr(kernel.place().layout) << " "
          << TargetToStr(kernel.place().target);
  VLOG(4) << "kernel.op_type():" << kernel.op_type();
  VLOG(4) << "kernel picker factors:" << kernel_pick_factors;
  VLOG(4) << "kernel place:" << kernel.place().DebugString();
  VLOG(4) << "winner_picker place:" << winner_place.DebugString();
  VLOG(4) << "------------------------------";

  // The data layout is not considered, for the input and output arguments
  // might have different data layout.
  // TODO(Superjomn) reconsider the idea of taking the data layout as a kernel
  // specification.
  return final_score;
}

}  // namespace lite
}  // namespace paddle
