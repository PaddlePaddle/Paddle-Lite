/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "acl_tensor.h"

namespace paddle_mobile {
namespace operators {
namespace acl {

#ifdef USE_ACL
template <typename TensorType>
std::unique_ptr<arm_compute::ITensor> initialise_tensor(
    arm_compute::TensorInfo &info) {
  auto tensor = cpp14::make_unique<TensorType>();
  tensor->allocator()->init(info);
  return std::move(tensor);
}

template <typename TensorType>
void tensor_allocate(arm_compute::ITensor &tensor) {
  auto itensor = dynamic_cast<TensorType *>(&tensor);
  itensor->allocator()->allocate();
}

Tensor::Tensor(arm_compute::TensorInfo &info) noexcept
    : _target(TargetHint::DONT_CARE), _info(info), _tensor(nullptr) {}

Tensor::Tensor(Tensor &&src) noexcept
    : _target(src._target),
      _info(std::move(src._info)),
      _tensor(std::move(src._tensor)) {}

arm_compute::ITensor *Tensor::set_target(TargetHint target) {
  switch (target) {
#ifdef USE_OPENCL
    case TargetHint::OPENCL:
      _tensor = initialise_tensor<arm_compute::CLTensor>(_info);
      break;
#elif defined(USE_OPENGLES)
    case TargetHint::OPENGLES:
      _tensor = initialise_tensor<arm_compute::GCTensor>(_info);
      break;
#endif
    case TargetHint::NEON:
      _tensor = initialise_tensor<arm_compute::Tensor>(_info);
      break;
    default:
      break;
  }
  _target = target;
  return _tensor.get();
}

void Tensor::allocate() {
  switch (_target) {
#ifdef USE_OPENCL
    case TargetHint::OPENCL:
      tensor_allocate<arm_compute::CLTensor>(*_tensor);
      break;
#elif defined(USE_OPENGLES)
    case TargetHint::OPENGLES:
      tensor_allocate<arm_compute::GCTensor>(*_tensor);
      break;
#endif
    case TargetHint::NEON:
      tensor_allocate<arm_compute::Tensor>(*_tensor);
      break;
    default:
      break;
  }
}
void Tensor::map(bool blocking) {
#ifdef USE_OPENCL
  if (_target == TargetHint::OPENCL)
    dynamic_cast<arm_compute::CLTensor *>(tensor())->map(blocking);
#elif defined(USE_OPENGLES)
  if (_target == TargetHint::OPENGLES)
    dynamic_cast<arm_compute::GCTensor *>(tensor())->map(blocking);
#endif
}
void Tensor::unmap() {
#ifdef USE_OPENCL
  if (_target == TargetHint::OPENCL)
    dynamic_cast<arm_compute::CLTensor *>(tensor())->unmap();
#elif defined(USE_OPENGLES)
  if (_target == TargetHint::OPENGLES)
    dynamic_cast<arm_compute::GCTensor *>(tensor())->unmap();
#endif
}

template <typename SubTensorType, typename ParentTensorType>
std::unique_ptr<arm_compute::ITensor> initialise_subtensor(
    arm_compute::ITensor *parent, arm_compute::TensorShape shape,
    arm_compute::Coordinates coords) {
  auto ptensor = dynamic_cast<ParentTensorType *>(parent);
  auto subtensor = cpp14::make_unique<SubTensorType>(ptensor, shape, coords);
  return std::move(subtensor);
}
SubTensor::SubTensor(Tensor *parent, arm_compute::TensorShape &tensor_shape,
                     arm_compute::Coordinates &coords) noexcept
    : _target(TargetHint::DONT_CARE),
      _tensor_shape(tensor_shape),
      _coords(coords),
      _parent(nullptr),
      _subtensor(nullptr) {
  _parent = parent->tensor();
  _target = parent->target();

  instantiate_subtensor();
}
arm_compute::ITensor *SubTensor::set_target(TargetHint target) {
  return (target == _target) ? _subtensor.get() : nullptr;
}

arm_compute::ITensor *SubTensor::tensor() { return _subtensor.get(); }

const arm_compute::ITensor *SubTensor::tensor() const {
  return _subtensor.get();
}

TargetHint SubTensor::target() const { return _target; }

void SubTensor::allocate() {
  // NOP for sub-tensors
}

void SubTensor::instantiate_subtensor() {
  switch (_target) {
#ifdef USE_OPENCL
    case TargetHint::OPENCL:
      _subtensor = initialise_subtensor<arm_compute::CLSubTensor,
                                        arm_compute::ICLTensor>(
          _parent, _tensor_shape, _coords);
      break;
#endif
    default:
    case TargetHint::NEON:
      _subtensor =
          initialise_subtensor<arm_compute::SubTensor, arm_compute::ITensor>(
              _parent, _tensor_shape, _coords);
      break;
  }
}

#endif

}  // namespace acl
}  // namespace operators
}  // namespace paddle_mobile
