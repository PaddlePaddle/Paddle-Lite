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

#ifndef ACL_TENSOR_H_
#define ACL_TENSOR_H_

#ifdef USE_ACL
#ifdef USE_OPENCL
#include "arm_compute/runtime/CL/CLSubTensor.h"
#include "arm_compute/runtime/CL/CLTensor.h"
#elif defined(USE_OPENGLES)
#include "arm_compute/runtime/GLES_COMPUTE/GCTensor.h"
#endif
#include "arm_compute/runtime/SubTensor.h"
#include "arm_compute/runtime/Tensor.h"

#include <memory>

namespace paddle_mobile {
namespace operators {
namespace acl {
enum class TargetHint {
  DONT_CARE,
  OPENCL,
  OPENGLES,
  NEON,
};

enum class ConvolutionMethodHint {
  GEMM,
  DIRECT,
};
namespace cpp14 {
template <class T>
struct _Unique_if {
  typedef std::unique_ptr<T> _Single_object;
};

template <class T>
struct _Unique_if<T[]> {
  typedef std::unique_ptr<T[]> _Unknown_bound;
};

template <class T, size_t N>
struct _Unique_if<T[N]> {
  typedef void _Known_bound;
};

template <class T, class... Args>
typename _Unique_if<T>::_Single_object make_unique(Args &&... args) {
  return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

template <class T>
typename _Unique_if<T>::_Unknown_bound make_unique(size_t n) {
  typedef typename std::remove_extent<T>::type U;
  return std::unique_ptr<T>(new U[n]());
}

template <class T, class... Args>
typename _Unique_if<T>::_Known_bound make_unique(Args &&...);
}  // namespace cpp14

class Tensor {
 public:
  explicit Tensor(arm_compute::TensorInfo &info) noexcept;
  virtual ~Tensor() {}
  Tensor(Tensor &&src) noexcept;
  void set_info(arm_compute::TensorInfo &&info) { _info = info; }
  arm_compute::ITensor *set_target(TargetHint target);
  const arm_compute::TensorInfo &info() const { return _info; }
  arm_compute::ITensor *tensor() { return _tensor.get(); }
  void allocate();
  void init() {}
  TargetHint target() const { return _target; }
  virtual void map(bool blocking = true);
  virtual void unmap();

 private:
  TargetHint _target;
  arm_compute::TensorInfo _info;
  std::unique_ptr<arm_compute::ITensor> _tensor;
};

class SubTensor {
 public:
  SubTensor(Tensor *parent, arm_compute::TensorShape &tensor_shape,
            arm_compute::Coordinates &coords) noexcept;
  ~SubTensor() {}
  arm_compute::ITensor *tensor();
  const arm_compute::ITensor *tensor() const;
  TargetHint target() const;
  void allocate();
  arm_compute::ITensor *set_target(TargetHint target);

 private:
  /** Instantiates a sub-tensor */
  void instantiate_subtensor();

 private:
  /**< Target that this tensor is pinned on */
  TargetHint _target;
  /**< SubTensor shape */
  arm_compute::TensorShape _tensor_shape;
  /**< SubTensor Coordinates */
  arm_compute::Coordinates _coords;
  /**< Parent tensor */
  arm_compute::ITensor *_parent;
  /**< SubTensor */
  std::unique_ptr<arm_compute::ITensor> _subtensor;
};

}  // namespace acl
}  // namespace operators
}  // namespace paddle_mobile
#endif
#endif  // ACL_TENSOR_H_
