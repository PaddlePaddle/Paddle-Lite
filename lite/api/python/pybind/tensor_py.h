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

#ifndef LITE_API_PYTHON_PYBIND_TENSOR_PY_H_  // NOLINT
#define LITE_API_PYTHON_PYBIND_TENSOR_PY_H_
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cstring>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>
#include "lite/api/paddle_api.h"
#include "lite/api/python/pybind/pybind.h"
#include "lite/core/tensor.h"

namespace py = pybind11;

namespace paddle {
namespace lite {
namespace pybind {

using lite_api::PrecisionType;
using lite_api::TargetType;
using lite_api::Tensor;

////////////////////////////////////////////////////////////////
// Function Name: TensorDTypeToPyDTypeStr
// Usage: Transform Lite PresionType name into corresponding
//        numpy name.
////////////////////////////////////////////////////////////////
inline std::string TensorDTypeToPyDTypeStr(PrecisionType type) {
#define TENSOR_DTYPE_TO_PY_DTYPE(T, proto_type)  \
  if (type == proto_type) {                      \
    if (proto_type == PrecisionType::kFP16) {    \
      return "e";                                \
    } else {                                     \
      return py::format_descriptor<T>::format(); \
    }                                            \
  }

  TENSOR_DTYPE_TO_PY_DTYPE(float, PrecisionType::kFloat)
  TENSOR_DTYPE_TO_PY_DTYPE(double, PrecisionType::kFP64)
  TENSOR_DTYPE_TO_PY_DTYPE(float, PrecisionType::kFP16)
  TENSOR_DTYPE_TO_PY_DTYPE(bool, PrecisionType::kBool)

  TENSOR_DTYPE_TO_PY_DTYPE(uint8_t, PrecisionType::kUInt8)
  TENSOR_DTYPE_TO_PY_DTYPE(int8_t, PrecisionType::kInt8)
  TENSOR_DTYPE_TO_PY_DTYPE(int32_t, PrecisionType::kInt32)
  TENSOR_DTYPE_TO_PY_DTYPE(int64_t, PrecisionType::kInt64)
  TENSOR_DTYPE_TO_PY_DTYPE(int16_t, PrecisionType::kInt16)

#undef TENSOR_DTYPE_TO_PY_DTYPE
  LOG(FATAL) << "Error: Unsupported tensor data type!";
  return "";
}

////////////////////////////////////////////////////////////////
// Function Name: TensorToPyArray
// Usage: Transform tensor's data into numpy array
////////////////////////////////////////////////////////////////
inline py::array TensorToPyArray(const Tensor &tensor,
                                 bool need_deep_copy = false) {
  const auto &tensor_dims = tensor.shape();
  auto tensor_dtype = tensor.precision();
  size_t sizeof_dtype = lite_api::PrecisionTypeLength(tensor_dtype);
  std::vector<size_t> py_dims(tensor_dims.size());
  std::vector<size_t> py_strides(tensor_dims.size());

  size_t numel = 1;
  for (int i = tensor_dims.size() - 1; i >= 0; --i) {
    py_dims[i] = (size_t)tensor_dims[i];
    py_strides[i] = sizeof_dtype * numel;
    numel *= py_dims[i];
  }
  std::string py_dtype_str = TensorDTypeToPyDTypeStr(tensor.precision());

  if (!tensor.IsInitialized()) {
    return py::array(py::dtype(py_dtype_str.c_str()), py_dims);
  }

  const void *tensor_buf_ptr = static_cast<const void *>(tensor.data<int8_t>());
  auto base = py::cast(std::move(tensor));
  return py::array(py::dtype(py_dtype_str.c_str()),
                   py_dims,
                   py_strides,
                   const_cast<void *>(tensor_buf_ptr),
                   base);
}

////////////////////////////////////////////////////////////////
// Function Name: SetTensorFromPyArrayT
// Usage: Transform numpy of specified precision into tensor
////////////////////////////////////////////////////////////////
template <typename T>
void SetTensorFromPyArrayT(
    Tensor *self,
    const py::array_t<T, py::array::c_style | py::array::forcecast> &array,
    const TargetType &place) {
  std::vector<int64_t> dims;
  dims.reserve(array.ndim());
  for (decltype(array.ndim()) i = 0; i < array.ndim(); ++i) {
    dims.push_back(static_cast<int>(array.shape()[i]));
  }
  self->Resize(dims);

  auto dst = self->mutable_data<T>(place);
  std::memcpy(dst, array.data(), array.nbytes());
}

////////////////////////////////////////////////////////////////
// Function Name: SetTensorFromPyArrayT
// Usage: Create a tensor from input numpy array
// Todo: float16 and uint16_t inputs are not supported on
//       Paddle-Lite, while these two precision type are supported
//       on PaddlePaddle.
////////////////////////////////////////////////////////////////
void SetTensorFromPyArray(Tensor *self,
                          const py::object &obj,
                          const TargetType &place) {
  auto array = obj.cast<py::array>();
  if (py::isinstance<py::array_t<float>>(array)) {
    SetTensorFromPyArrayT<float>(self, array, place);
  } else if (py::isinstance<py::array_t<int>>(array)) {
    SetTensorFromPyArrayT<int>(self, array, place);
  } else if (py::isinstance<py::array_t<int64_t>>(array)) {
    SetTensorFromPyArrayT<int64_t>(self, array, place);
  } else if (py::isinstance<py::array_t<double>>(array)) {
    SetTensorFromPyArrayT<double>(self, array, place);
  } else if (py::isinstance<py::array_t<int8_t>>(array)) {
    SetTensorFromPyArrayT<int8_t>(self, array, place);
  } else if (py::isinstance<py::array_t<int16_t>>(array)) {
    SetTensorFromPyArrayT<int16_t>(self, array, place);
  } else if (py::isinstance<py::array_t<uint8_t>>(array)) {
    SetTensorFromPyArrayT<uint8_t>(self, array, place);
  } else if (py::isinstance<py::array_t<bool>>(array)) {
    SetTensorFromPyArrayT<bool>(self, array, place);
  } else {
    // obj may be any type, obj.cast<py::array>() may be failed,
    // then the array.dtype will be string of unknown meaning,
    LOG(FATAL) << "Input object type error or incompatible array data type. "
                  "tensor.from_numpy(numpy.array, PrecisionType) supports "
                  "numpy array input in  bool, float32, "
                  "float64, int8, int16, int32, int64 or uint8, please check "
                  "your input or input array data type.";
  }
}

}  // namespace pybind
}  // namespace lite
}  // namespace paddle

#endif  // NOLINT
